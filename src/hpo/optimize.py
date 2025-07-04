import torch
import torch.nn as nn
import torch.optim as optim
from ray import tune
from ray.tune import Trainable
import multiprocessing as mp
import os
import gc
from ray.tune.search.optuna import OptunaSearch
from ray.air import RunConfig
# ### 추가: 자동 혼합 정밀도(AMP)를 위한 import ###
from torch.cuda.amp import autocast, GradScaler

from src.pid_training.pid_net import PIDNet
from src.pid_training.plant import FirstOrderPlant
from src.utils.loss import GeometricLoss

def calculate_flow_divergence(flow_vectors: torch.Tensor, target_vectors: torch.Tensor) -> torch.Tensor:
    actual_flows = flow_vectors[1:] - flow_vectors[:-1]
    ideal_flows = target_vectors[1:] - flow_vectors[:-1]
    
    norm_actual = torch.linalg.norm(actual_flows, dim=-1)
    norm_ideal = torch.linalg.norm(ideal_flows, dim=-1)
    
    dot_product = torch.sum(actual_flows * ideal_flows, dim=-1)
    cosine_sim = dot_product / (norm_actual * norm_ideal + 1e-8)
    
    divergence = 1 - cosine_sim.mean()
    return divergence if not torch.isnan(divergence) else torch.tensor(0.0, device=flow_vectors.device)


class PIDTrainable(Trainable):
    def setup(self, config):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.config = config
        
        self.pid_net = PIDNet().to(self.device)
        self.plant = FirstOrderPlant(ku=2.0, tau=5.0)
        
        self.optimizer = optim.Adam(self.pid_net.parameters(), lr=config["lr"])
        self.criterion = nn.MSELoss()
        
        self.geometric_loss_calculator = GeometricLoss(alpha=config["alpha"], gamma=0.0).to(self.device)

        self.dt = torch.tensor(0.1, device=self.device)
        self.setpoint = torch.tensor(1.0, device=self.device)

        # ### 추가: GradScaler 초기화 ###
        # autocast에서 float16으로 계산 시 그래디언트가 너무 작아지는 현상(underflow)을 방지
        self.scaler = GradScaler()

    def step(self):
        self.pid_net.train()
        
        self.plant.reset()
        self.pid_net.reset()
        
        plant_outputs, flow_vectors_list = [], []
        current_value = self.plant.state.clone().to(self.device)

        # ### 수정: 시뮬레이션 스텝 복원 ###
        sim_steps = 200
        for _ in range(sim_steps):
            # ### autocast 컨텍스트 추가 ###
            # 이 블록 내의 연산들은 A100에 최적화된 float16으로 자동 변환되어 실행됩니다.
            with autocast():
                control_input = self.pid_net(self.setpoint, current_value, self.dt.item())
                current_value = self.plant.step(control_input.to('cpu'), self.dt.to('cpu')).to(self.device)
            
            plant_outputs.append(current_value)
            flow_vectors_list.append(control_input)

        with autocast():
            all_plant_outputs = torch.stack(plant_outputs)
            all_setpoints = self.setpoint.expand_as(all_plant_outputs)
            main_loss = self.criterion(all_plant_outputs, all_setpoints)

            flow_vectors = torch.stack(flow_vectors_list)
            ideal_control_value = self.setpoint.item() / self.plant.ku.item()
            target_vectors = torch.full_like(flow_vectors, fill_value=ideal_control_value)

            resonance_loss_val = self.geometric_loss_calculator.calculate_resonance_loss(
                flow_vectors.unsqueeze(0),
                target_vectors.unsqueeze(0)
            )
            
            flow_divergence = calculate_flow_divergence(flow_vectors, target_vectors)

            final_loss = main_loss + self.config["alpha"] * resonance_loss_val + self.config["beta"] * flow_divergence

        self.optimizer.zero_grad(set_to_none=True) # set_to_none=True는 약간의 성능 향상

        # ### 수정: scaler를 사용하여 역전파 및 스텝 실행 ###
        self.scaler.scale(final_loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        
        return {
            "loss": final_loss.item(),
            "main_loss": main_loss.item(),
            "divergence": flow_divergence.item(),
        }
    
    def save_checkpoint(self, checkpoint_dir):
        checkpoint_path = os.path.join(checkpoint_dir, "model.pth")
        torch.save(self.pid_net.state_dict(), checkpoint_path)
        return checkpoint_dir

    def load_checkpoint(self, checkpoint_dir):
        checkpoint_path = os.path.join(checkpoint_dir, "model.pth")
        self.pid_net.load_state_dict(torch.load(checkpoint_path))

    def cleanup(self):
        del self.pid_net, self.plant, self.optimizer, self.geometric_loss_calculator, self.scaler
        gc.collect()
        if self.device == "cuda":
            torch.cuda.empty_cache()

def main():
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    search_space = {
        "lr": tune.loguniform(1e-4, 1e-1),
        "alpha": tune.uniform(0.0, 0.5),
        "beta": tune.uniform(0.0, 0.5),
    }
    
    search_alg = OptunaSearch(metric="loss", mode="min")
    
    tuner = tune.Tuner(
        tune.with_resources(PIDTrainable, {"cpu": 1, "gpu": 1}),
        param_space=search_space,
        tune_config=tune.TuneConfig(
            search_alg=search_alg, 
            num_samples=20,
            # ### 수정: 동시 실행 Trial 수 상향 ###
            max_concurrent_trials=4 
        ),
        run_config=RunConfig(
            name="Geometric_PID_HPO_A100",
            stop={"training_iteration": 10},
        ),
    )
    
    print("Phase 2 HPO Started on A100: Searching for optimal geometric parameters with AMP...")
    results = tuner.fit()
    
    best_result = results.get_best_result("loss", "min")

    print("\n--- HPO Finished ---")
    print(f"Best trial final validation loss: {best_result.metrics['loss']:.4f}")
    print("Best hyperparameters found were: ", best_result.config)
    print("--------------------")

if __name__ == '__main__':
    main()