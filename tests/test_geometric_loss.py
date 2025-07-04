import torch
import torch.nn.functional as F
import pytest
from src.utils.loss import GeometricLoss

# 테스트에 사용할 고정된 텐서 fixture 생성
@pytest.fixture
def sample_tensors():
    """테스트를 위한 샘플 텐서를 제공합니다."""
    # (batch_size, sequence_length, d_model)
    # 2개의 배치, 각 3개의 토큰, 4차원 벡터
    batch_size = 2
    seq_len = 3
    d_model = 4
    
    # 모델의 최종 출력 (의미 흐름의 결과)
    # 두 배치의 방향성이 다르게 설정
    # 배치 1: [1, 1, 1, 1] 방향
    # 배치 2: [-1, -1, -1, -1] 방향
    flow_vectors = torch.tensor([
        [[1.0, 1.0, 1.0, 1.0], [1.1, 1.1, 1.1, 1.1], [0.9, 0.9, 0.9, 0.9]],
        [[-1.0, -1.0, -1.0, -1.0], [-1.1, -1.1, -1.1, -1.1], [-0.9, -0.9, -0.9, -0.9]]
    ], dtype=torch.float32)

    # 정답 토큰의 이상적인 임베딩 벡터
    # 배치 1: 모델 출력과 같은 방향
    # 배치 2: 모델 출력과 반대 방향
    target_vectors = torch.tensor([
        [[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]],
        [[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]]
    ], dtype=torch.float32)

    return flow_vectors, target_vectors

def test_resonance_loss_calculation(sample_tensors):
    """
    Resonance Loss (공명 손실)가 올바르게 계산되는지 검증합니다.
    - 두 벡터의 방향이 같으면 Loss는 음수 (보상).
    - 두 벡터의 방향이 반대면 Loss는 양수 (페널티).
    """
    flow_vectors, target_vectors = sample_tensors
    geometric_loss = GeometricLoss(alpha=1.0, gamma=0.0) # 공명 손실만 테스트
    
    resonance_loss = geometric_loss.calculate_resonance_loss(flow_vectors, target_vectors)

    # 코사인 유사도 계산
    # 배치 1의 유사도는 1에 가까워야 함 (방향 일치)
    # 배치 2의 유사도는 -1에 가까워야 함 (방향 반대)
    cosine_sim = F.cosine_similarity(flow_vectors, target_vectors, dim=-1)
    
    # 공명 손실은 -E[cos(sim)] 이므로, 기댓값은 -mean(cosine_sim)
    expected_loss = -torch.mean(cosine_sim)

    assert torch.isclose(resonance_loss, expected_loss), \
        f"계산된 공명 손실({resonance_loss})이 예상값({expected_loss})과 다릅니다."

def test_curvature_regularization_calculation():
    """
    Curvature Regularization (곡률 정칙화) 항이 올바르게 계산되는지 검증합니다.
    (Finite Difference Method 근사)
    """
    d_model = 4
    # 중앙, 이전, 다음 시점의 벡터
    x_curr = torch.randn(1, 1, d_model)
    x_prev = torch.randn(1, 1, d_model)
    x_next = torch.randn(1, 1, d_model)
    
    geometric_loss = GeometricLoss(alpha=0.0, gamma=1.0) # 곡률 손실만 테스트

    # 2차 미분 근사 (x_next - 2*x_curr + x_prev)
    curvature_loss = geometric_loss.calculate_curvature_loss(x_curr, x_prev, x_next)
    
    second_derivative_approx = x_next + x_prev - 2 * x_curr
    expected_loss = torch.norm(second_derivative_approx)

    assert torch.isclose(curvature_loss, expected_loss), \
        f"계산된 곡률 손실({curvature_loss})이 예상값({expected_loss})과 다릅니다."