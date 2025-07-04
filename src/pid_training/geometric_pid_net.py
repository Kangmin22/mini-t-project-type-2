import torch
import torch.nn as nn

class GeometricPIDNet(nn.Module):
    """
    고차원 벡터 공간에서 작동하는 기하학적 PID 제어기.
    Kp, Ki, Kd 게인을 스칼라가 아닌, d_model 차원의 벡터를 변환하는
    학습 가능한 행렬(nn.Linear)로 구현합니다.
    """
    def __init__(self, d_model: int):
        """
        Args:
            d_model (int): 의미 벡터의 차원.
        """
        super().__init__()
        self.d_model = d_model

        # Kp, Ki, Kd를 d_model x d_model 크기의 학습 가능한 행렬로 선언합니다.
        # bias=False는 순수한 선형 변환(회전/스케일링)에 집중하기 위함입니다.
        self.Kp = nn.Linear(d_model, d_model, bias=False)
        self.Ki = nn.Linear(d_model, d_model, bias=False)
        self.Kd = nn.Linear(d_model, d_model, bias=False)

        # 내부 상태(버퍼)들도 이제 d_model 차원의 벡터가 됩니다.
        self.register_buffer('integral_term', torch.zeros(d_model))
        self.register_buffer('last_error_vector', torch.zeros(d_model))

    def forward(self, current_vector: torch.Tensor, target_vector: torch.Tensor, dt: float) -> torch.Tensor:
        """
        오차 '벡터'를 입력받아 보정 '벡터'를 출력합니다.

        Args:
            current_vector (torch.Tensor): 현재 상태 벡터. (Batch, d_model)
            target_vector (torch.Tensor): 목표 상태 벡터. (Batch, d_model)
            dt (float): 시간 간격.

        Returns:
            torch.Tensor: 계산된 보정 벡터. (Batch, d_model)
        """
        error_vector = target_vector - current_vector

        # 적분항 업데이트 (벡터 연산)
        # 이전 기록을 분리하여 그래디언트가 과도하게 길어지는 것을 방지합니다.
        self.integral_term = self.integral_term.detach() + error_vector * dt
        
        # 미분항 계산 (벡터 연산)
        derivative_vector = (error_vector - self.last_error_vector) / dt

        # 각 항은 이제 벡터에 대한 선형 변환으로 계산됩니다.
        p_correction = self.Kp(error_vector)
        i_correction = self.Ki(self.integral_term)
        d_correction = self.Kd(derivative_vector)

        # 다음 스텝을 위해 현재 오차 벡터를 그래디언트 흐름 없이 저장합니다.
        self.last_error_vector = error_vector.detach()

        # 최종 보정 벡터는 각 보정 벡터의 합입니다.
        correction_vector = p_correction + i_correction + d_correction
        
        return correction_vector

    def reset(self):
        """시뮬레이션 에피소드 시작 시 내부 상태 벡터를 초기화합니다."""
        self.integral_term.zero_()
        self.last_error_vector.zero_()