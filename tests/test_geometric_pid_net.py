import torch
import torch.nn as nn
import pytest

# 아직 존재하지 않지만, 우리가 곧 만들 클래스를 import 합니다.
from src.pid_training.geometric_pid_net import GeometricPIDNet

@pytest.fixture
def geometric_pid_config():
    """테스트를 위한 기본 설정을 제공합니다."""
    return {
        "d_model": 128, # 의미 벡터의 차원
        "dt": 0.1
    }

def test_geometric_pid_net_initialization(geometric_pid_config):
    """GeometricPIDNet이 올바르게 초기화되는지 검증합니다."""
    d_model = geometric_pid_config["d_model"]
    gpid_net = GeometricPIDNet(d_model=d_model)

    # Kp, Ki, Kd가 스칼라가 아닌 nn.Linear 레이어인지 확인
    assert isinstance(gpid_net.Kp, nn.Linear), "Kp는 nn.Linear 레이어야 합니다."
    assert isinstance(gpid_net.Ki, nn.Linear), "Ki는 nn.Linear 레이어야 합니다."
    assert isinstance(gpid_net.Kd, nn.Linear), "Kd는 nn.Linear 레이어야 합니다."
    
    # 입출력 차원이 d_model과 일치하는지 확인
    assert gpid_net.Kp.in_features == d_model
    assert gpid_net.Kp.out_features == d_model

def test_geometric_pid_net_forward_pass(geometric_pid_config):
    """GeometricPIDNet의 forward pass가 올바른 형태의 출력 벡터를 반환하는지 검증합니다."""
    d_model = geometric_pid_config["d_model"]
    dt = geometric_pid_config["dt"]
    gpid_net = GeometricPIDNet(d_model=d_model)

    batch_size = 4
    # (batch_size, d_model) 형태의 임의의 텐서 생성
    current_vector = torch.randn(batch_size, d_model)
    target_vector = torch.randn(batch_size, d_model)

    # forward pass 실행
    correction_vector = gpid_net(current_vector, target_vector, dt)

    # 출력 벡터의 형태가 (batch_size, d_model)인지 확인
    assert correction_vector.shape == (batch_size, d_model), \
        f"출력 벡터의 형태가 올바르지 않습니다. 예상: {(batch_size, d_model)}, 실제: {correction_vector.shape}"

    # 출력 벡터가 그래디언트를 필요로 하는지 확인 (학습 가능해야 함)
    assert correction_vector.requires_grad, "출력 벡터는 그래디언트 계산이 가능해야 합니다."