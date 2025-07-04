import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.pid_training.pid_net import PIDNet
from src.pid_training.plant import FirstOrderPlant
from src.utils.loss import GeometricLoss

def main():
    """
    GeometricLoss를 적용하여 PIDNet을 학습시키는 메인 함수.
    """
    # --- 1. 설정 (Configuration) ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    pid_net = PIDNet().to(device)
    plant = FirstOrderPlant(ku=2.0, tau=5.0).to(device)
    criterion = nn.MSELoss()
    geometric_loss_calculator = GeometricLoss(alpha=0.1, gamma=0.0).to(device)
    optimizer = optim.Adam(pid_net.parameters(), lr=0.01)

    num_epochs = 100
    sim_steps = 200
    dt = torch.tensor(0.1, device=device)
    setpoint = torch.tensor(1.0, device=device)

    epoch_losses = []

    print("Phase 1 Training Started: PIDNet with GeometricLoss")
    print(f"Hyperparameters: alpha={geometric_loss_calculator.alpha}, gamma={geometric_loss_calculator.gamma}")

    # --- 2. 학습 루프 (Training Loop) ---
    for epoch in tqdm(range(num_epochs), desc="Training Progress"):
        plant.reset()
        pid_net.reset()
        optimizer.zero_grad()
        
        # ### 수정된 부분 ###
        # 시뮬레이션 결과를 저장할 리스트
        plant_outputs = []
        flow_vectors_list = []

        current_value = plant.state.clone()

        for _ in range(sim_steps):
            control_input = pid_net(setpoint, current_value, dt.item())
            current_value = plant.step(control_input, dt)

            # Loss를 바로 계산하지 않고, 결과를 리스트에 저장
            plant_outputs.append(current_value)
            flow_vectors_list.append(control_input.unsqueeze(0))

        # --- 3. 최종 Loss 계산 ---
        # 루프가 끝난 후, 모든 출력을 하나의 텐서로 합침
        all_plant_outputs = torch.cat(plant_outputs)
        
        # 목표값도 출력과 같은 모양으로 만듦
        all_setpoints = setpoint.expand_as(all_plant_outputs)
        
        # 이제 Loss를 딱 한 번만 계산!
        main_loss = criterion(all_plant_outputs, all_setpoints)
        
        # Resonance Loss 계산
        ideal_control_value = setpoint.item() / plant.ku.item()
        # 제어 입력의 최종 목표값은 스칼라이므로 unsqueeze로 차원을 맞춰줌
        target_control_vector = torch.full_like(control_input, fill_value=ideal_control_value) 
        
        # `flow_vectors`도 하나의 텐서로 합침
        flow_vectors = torch.cat(flow_vectors_list, dim=0).unsqueeze(1)
        target_vectors = target_control_vector.expand_as(flow_vectors)
        
        final_loss = geometric_loss_calculator(
            main_loss=main_loss,
            model_outputs={'hidden_states': [flow_vectors]},
            target_vectors=target_vectors
        )

        final_loss.backward()
        optimizer.step()
        epoch_losses.append(final_loss.item())

    print("\nTraining Finished.")
    print(f"Final Loss: {epoch_losses[-1]:.4f}")
    print(f"Trained PID Gains: Kp={pid_net.Kp.item():.4f}, Ki={pid_net.Ki.item():.4f}, Kd={pid_net.Kd.item():.4f}")

    # --- 4. 결과 시각화 ---
    plt.figure(figsize=(10, 5))
    plt.plot(epoch_losses)
    plt.title("Training Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.savefig("phase1_loss_curve.png")
    print("Loss curve saved to 'phase1_loss_curve.png'")


if __name__ == "__main__":
    main()