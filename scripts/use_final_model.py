# FILE: scripts/use_final_model.py
import torch
import time

from src.pid_training.pid_net import PIDNet
from src.pid_training.plant import FirstOrderPlant

def run_inference():
    """
    저장된 최종 PIDNet 모델을 불러와 제어 시뮬레이션을 실행합니다.
    """
    # HPO에서 찾았던 최적 파라미터 값으로 모델 구조를 초기화합니다.
    # 이 값들은 모델의 '뼈대'를 정의하며, 실제 가중치는 파일에서 불러옵니다.
    BEST_CONFIG = {
        "kp": 4.8215,
        "ki": 0.6817,
        "kd": 1.1960
    }
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"--- 추론 시뮬레이션을 시작합니다. (Device: {device}) ---")
    
    # 1. 모델 뼈대 생성
    model = PIDNet(kp=BEST_CONFIG["kp"], ki=BEST_CONFIG["ki"], kd=BEST_CONFIG["kd"]).to(device)
    
    # 2. 학습된 가중치 불러오기
    model_path = "final_pid_net.pth"
    model.load_state_dict(torch.load(model_path))
    
    # 3. 추론 모드로 설정
    model.eval()

    # 시뮬레이션 시작
    plant = FirstOrderPlant(ku=2.0, tau=5.0, device=device)
    current_val = torch.tensor(0.0).to(device)
    target_val = torch.tensor(10.0).to(device) # 목표: 10
    dt = torch.tensor(0.1).to(device)

    print("\n[추론 시뮬레이션 시작]")
    print(f"목표 값: {target_val.item():.2f}")
    print("-------------------------")

    with torch.no_grad(): # 추론 시에는 그래디언트 계산이 필요 없습니다.
        for i in range(50):
            control_signal = model(current_val, target_val, dt)
            current_val = plant.step(control_signal, dt)
            print(f"스텝 {i+1:02d}: 현재 값 = {current_val.item():.4f}")
            time.sleep(0.05)
            
    print("-------------------------")
    print(f"최종 값: {current_val.item():.4f}")
    print("[추론 시뮬레이션 종료]")


if __name__ == '__main__':
    run_inference()