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

    def step(self):
        self.pid_net.train()
        
        self.plant.reset()
        self.pid_net.reset()
        
        plant_outputs, flow_vectors_list = [], []
        current_value = self.plant.state.clone().to(self.device)

        sim_steps = 50 # 메모리 사용량 감소
        for _ in range(sim_steps):
            control_input = self.pid_net(self.setpoint, current_value, self.dt.item())
            current_value = self.plant.step(control_input.to('cpu'), self.dt.to('cpu')).to(self.device)
            
            plant_outputs.append(current_value)
            flow_vectors_list.append(control_input)

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

        self.optimizer.zero_grad()
        final_loss.backward()
        self.optimizer.step()
        
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
        del self.pid_net, self.plant, self.optimizer, self.geometric_loss_calculator
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
            max_concurrent_trials=1 # 동시 실행 1개로 제한
        ),
        run_config=RunConfig(
            name="Geometric_PID_HPO",
            stop={"training_iteration": 10},
        ),
    )
    
    print("Phase 2 HPO Started: Searching for optimal geometric parameters...")
    results = tuner.fit()
    
    best_result = results.get_best_result("loss", "min")

    print("\n--- HPO Finished ---")
    print(f"Best trial final validation loss: {best_result.metrics['loss']:.4f}")
    print("Best hyperparameters found were: ", best_result.config)
    print("--------------------")

if __name__ == '__main__':
    main()