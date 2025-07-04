import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
import json

from src.pid_training.pid_net import PIDNet
from src.pid_training.plant import FirstOrderPlant
from src.utils.loss import GeometricLoss

def main():
    """
    HPO로 찾은 최적의 하이퍼파라미터를 사용하여 최종 PIDNet 모델을 훈련하고 저장합니다.
    """
    # --- 1. 설정: HPO에서 찾은 새로운 최적의 값으로 업데이트! ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    BEST_HPARAMS = {
        'lr': 0.07973517675218665, 
        'alpha': 0.3462792257398627, 
        'beta': 0.08583256327599675
    }
    
    print("Training final model with best hyperparameters:")
    print(json.dumps(BEST_HPARAMS, indent=2))

    pid_net = PIDNet().to(device)
    plant = FirstOrderPlant(ku=2.0, tau=5.0)
    criterion = nn.MSELoss()
    geometric_loss_calculator = GeometricLoss(alpha=BEST_HPARAMS["alpha"], gamma=0.0).to(device)
    optimizer = optim.Adam(pid_net.parameters(), lr=BEST_HPARAMS["lr"])

    num_epochs = 300
    sim_steps = 200
    dt = torch.tensor(0.1, device=device)
    setpoint = torch.tensor(1.0, device=device)

    epoch_losses = []

    # --- 2. 학습 루프 ---
    for epoch in tqdm(range(num_epochs), desc="Training Final Model"):
        plant.reset()
        pid_net.reset()
        
        plant_outputs, flow_vectors_list = [], []
        current_value = plant.state.clone().to(device)

        for _ in range(sim_steps):
            control_input = pid_net(setpoint, current_value, dt.item())
            current_value = plant.step(control_input.to('cpu'), dt.to('cpu')).to(device)
            plant_outputs.append(current_value)
            flow_vectors_list.append(control_input)

        # --- 3. Loss 계산 및 역전파 ---
        all_plant_outputs = torch.stack(plant_outputs)
        all_setpoints = setpoint.expand_as(all_plant_outputs)
        main_loss = criterion(all_plant_outputs, all_setpoints)
        
        flow_vectors = torch.stack(flow_vectors_list)
        ideal_control_value = setpoint.item() / plant.ku.item()
        target_vectors = torch.full_like(flow_vectors, fill_value=ideal_control_value)

        resonance_loss_val = geometric_loss_calculator.calculate_resonance_loss(
            flow_vectors.unsqueeze(0), target_vectors.unsqueeze(0))
        
        # 최종 모델에서는 beta(경로 효율성)항도 최종 Loss에 반영해 줄 수 있습니다.
        # 하지만 단순화를 위해 현재는 alpha만 사용합니다.
        final_loss = main_loss + BEST_HPARAMS["alpha"] * resonance_loss_val
        
        optimizer.zero_grad()
        final_loss.backward()
        optimizer.step()
        epoch_losses.append(final_loss.item())

    print("\n--- Final Model Training Finished ---")
    print(f"Final Loss: {epoch_losses[-1]:.4f}")
    
    # --- 4. 최종 모델 저장 ---
    final_model_path = "final_pid_net.pth"
    torch.save(pid_net.state_dict(), final_model_path)
    print(f"Final model saved to '{final_model_path}'")

    # --- 5. 결과 시각화 ---
    plt.figure(figsize=(10, 5))
    plt.plot(epoch_losses)
    plt.title("Final Model Training Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.savefig("final_model_loss_curve.png")
    print("Loss curve saved to 'final_model_loss_curve.png'")


if __name__ == "__main__":
    main()