# FILE: tests/test_pid_net.py
import pytest
import torch
from src.pid_training.pid_net import PIDNet
from src.pid_training.plant import FirstOrderPlant
from src.pid_training.dataset import PIDDataset

def test_pid_net_output_shape():
    """PIDNet의 출력 형태를 검증합니다."""
    pid = PIDNet(kp=1.0, ki=1.0, kd=1.0)
    current = torch.tensor(5.0)
    target = torch.tensor(10.0)
    dt = torch.tensor(0.1)
    output = pid(current, target, dt)
    assert output.shape == torch.Size([]) # 스칼라 값이어야 함

def test_pid_net_learnable_params():
    """PID 게인 값들이 학습 가능한 파라미터인지 검증합니다."""
    pid = PIDNet(kp=1.0, ki=1.0, kd=1.0)
    params = list(pid.parameters())
    assert len(params) == 3
    for p in params:
        assert p.requires_grad

def test_pid_net_reset():
    """PIDNet의 리셋 기능이 내부 상태를 초기화하는지 검증합니다."""
    pid = PIDNet(kp=1.0, ki=1.0, kd=1.0)
    # 한 스텝 진행하여 내부 상태 변경
    pid(torch.tensor(5.0), torch.tensor(10.0), torch.tensor(0.1))
    
    # 내부 상태가 0이 아님을 확인
    assert pid.integral_term.item() != 0
    assert pid.prev_error.item() != 0

    # 리셋
    pid.reset()

    # 내부 상태가 0으로 초기화되었는지 확인
    assert pid.integral_term.item() == 0
    assert pid.prev_error.item() == 0

def test_first_order_plant_step():
    """FirstOrderPlant의 상태가 올바르게 업데이트되는지 검증합니다."""
    plant = FirstOrderPlant(ku=2.0, tau=5.0)
    initial_state = plant.state.clone()
    # 제어 입력을 1.0으로 0.1초간 가함
    new_state = plant.step(control_input=torch.tensor(1.0), dt=torch.tensor(0.1))
    
    # 상태가 변했는지 확인
    assert new_state != initial_state

def test_pid_dataset():
    """PIDDataset이 시뮬레이션 설정을 생성하는지 검증합니다."""
    dataset = PIDDataset(num_samples=10, steps_per_episode=100)
    assert len(dataset) == 10
    
    # 한 개의 샘플을 가져와서 구조 확인
    sample = dataset[0]
    assert "target_setpoints" in sample
    assert "plant_params" in sample
    assert sample["target_setpoints"].shape == (100,)
    assert "ku" in sample["plant_params"]
    assert "tau" in sample["plant_params"]