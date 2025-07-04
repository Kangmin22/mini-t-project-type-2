import torch
import torch.optim as optim
import torch.nn as nn

from src.pid_training.geometric_pid_net import GeometricPIDNet

def run_sanity_check():
    """
    GeometricPIDNet이 실제로 학습 가능한지(그래디언트가 흐르는지)
    확인하는 간단한 테스트 함수.
    """
    print("--- Starting GeometricPIDNet Sanity Check ---")

    # 1. 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d_model = 128
    batch_size = 4
    
    # 2. 모델 및 옵티마이저 초기화
    gpid_net = GeometricPIDNet(d_model=d_model).to(device)
    optimizer = optim.Adam(gpid_net.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    # 3. 더미 데이터 생성
    current_vector = torch.randn(batch_size, d_model).to(device)
    target_vector = torch.randn(batch_size, d_model).to(device)
    dt = 0.1

    # 4. Forward Pass -> Loss 계산 -> Backward Pass
    optimizer.zero_grad()
    
    # 모델로부터 보정 벡터를 얻음
    correction_vector = gpid_net(current_vector, target_vector, dt)
    
    # 가상의 플랜트: 현재 상태에 보정 벡터를 더한 것이 다음 상태라고 가정
    next_vector = current_vector + correction_vector
    
    # Loss 계산
    loss = criterion(next_vector, target_vector)
    
    # 역전파
    loss.backward()

    # 5. 그래디언트 확인
    # Kp 행렬의 가중치에 그래디언트가 계산되었는지 확인
    kp_grad = gpid_net.Kp.weight.grad
    
    if kp_grad is not None and kp_grad.abs().sum() > 0:
        print("\n✅ Sanity check PASSED!")
        print("Gradients are flowing correctly through the GeometricPIDNet.")
        print(f"Sum of absolute gradients in Kp weight: {kp_grad.abs().sum().item()}")
    else:
        print("\n❌ Sanity check FAILED!")
        print("Gradients are not flowing.")

    print("\n--- Sanity Check Finished ---")


if __name__ == "__main__":
    run_sanity_check()