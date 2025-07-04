# FILE: scripts/train_final_model.py
import torch
import torch.optim as optim
import os

from src.pid_training.pid_net import PIDNet
from src.pid_training.plant import FirstOrderPlant

def train_and_save_best_model():
    """
    HPO에서 찾은 최적의 하이퍼파라미터로 PIDNet을 학습하고 저장합니다.
    """
    # HPO 최종 결과에서 얻은 최적의 하이퍼파라미터
    BEST_CONFIG = {
        "lr": 0.0597,
        "kp": 4.8215,
        "ki": 0.6817,
        "kd": 1.1960
    }
    
    # GPU 설정
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"--- 최종 모델 학습을 시작합니다. (Device: {device}) ---")

    # 모델, 플랜트, 옵티마이저 초기화
    pid_net = PIDNet(kp=BEST_CONFIG["kp"], ki=BEST_CONFIG["ki"], kd=BEST_CONFIG["kd"]).to(device)
    plant = FirstOrderPlant(ku=2.0, tau=5.0, device=device)
    optimizer = optim.Adam(pid_net.parameters(), lr=BEST_CONFIG["lr"])
    
    # 학습 설정
    training_epochs = 20 # 충분히 많은 에포크로 학습
    steps_per_epoch = 200

    # 학습 루프
    for epoch in range(training_epochs):
        total_loss = 0.0
        current_val = torch.tensor(0.0).to(device)
        plant.reset()
        pid_net.reset()

        for _ in range(steps_per_epoch):
            target_val = torch.tensor(10.0).to(device)
            dt = torch.tensor(0.1).to(device)
            
            u = pid_net(current_val, target_val, dt)
            next_val = plant.step(u, dt)
            loss = (target_val - next_val) ** 2
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            current_val = next_val.detach()
            
        avg_loss = total_loss / steps_per_epoch
        print(f"Epoch [{epoch+1}/{training_epochs}], Loss: {avg_loss:.4f}")

    # 학습된 모델 저장
    save_path = "final_pid_net.pth"
    torch.save(pid_net.state_dict(), save_path)
    print(f"\n✅ 최종 학습된 모델을 '{save_path}'에 저장했습니다.")


if __name__ == '__main__':
    train_and_save_best_model()