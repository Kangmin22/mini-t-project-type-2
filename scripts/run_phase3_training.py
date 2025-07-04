import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.pid_training.geometric_pid_net import GeometricPIDNet
from src.pid_training.multi_dim_plant import MultiDimPlant

def main():
    """GeometricPIDNet을 훈련시키는 메인 스크립트."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Starting Phase 3 Training on device: {device} ---")

    # 설정
    d_model = 64 # 벡터 차원
    lr = 0.01
    num_epochs = 200
    sim_steps = 100
    dt = torch.tensor(0.1, device=device)

    # 모델, 플랜트, 옵티마이저 초기화
    gpid_net = GeometricPIDNet(d_model=d_model).to(device)
    plant = MultiDimPlant(d_model=d_model).to(device)
    optimizer = optim.Adam(gpid_net.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # 목표 상태 벡터 (모든 차원에서 1.0을 목표로 함)
    setpoint_vector = torch.ones(d_model, device=device)
    
    epoch_losses = []

    for epoch in tqdm(range(num_epochs), desc="Training GeometricPIDNet"):
        plant.reset()
        gpid_net.reset()
        optimizer.zero_grad()

        plant_outputs = []
        current_vector = plant.state.clone()

        for _ in range(sim_steps):
            correction_vector = gpid_net(current_vector, setpoint_vector, dt.item())
            
            # 제어 입력 = 현재 상태 + 보정 벡터
            control_input = current_vector + correction_vector
            
            next_vector = plant.step(control_input, dt)
            plant_outputs.append(next_vector)
            current_vector = next_vector

        all_outputs = torch.stack(plant_outputs)
        all_setpoints = setpoint_vector.unsqueeze(0).expand_as(all_outputs)
        
        loss = criterion(all_outputs, all_setpoints)
        loss.backward()
        optimizer.step()
        
        epoch_losses.append(loss.item())

    print("\n--- Training Finished ---")
    print(f"Final Loss: {epoch_losses[-1]:.6f}")

    # 최종 모델 저장
    final_model_path = "geometric_pid_net.pth"
    torch.save(gpid_net.state_dict(), final_model_path)
    print(f"Final geometric model saved to '{final_model_path}'")

    # 결과 시각화
    plt.figure(figsize=(10, 5))
    plt.plot(epoch_losses)
    plt.title("GeometricPIDNet Training Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.savefig("geometric_pid_loss_curve.png")
    print("Loss curve saved to 'geometric_pid_loss_curve.png'")

if __name__ == "__main__":
    main()