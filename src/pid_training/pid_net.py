import torch
import torch.nn as nn

class PIDNet(nn.Module):
    """
    학습 가능한 파라미터(Kp, Ki, Kd)를 가진 PID 제어기.
    """
    def __init__(self):
        super().__init__()
        self.Kp = nn.Parameter(torch.randn(1) * 0.1)
        self.Ki = nn.Parameter(torch.randn(1) * 0.01)
        self.Kd = nn.Parameter(torch.randn(1) * 0.01)

        self.register_buffer('integral_term', torch.zeros(1))
        self.register_buffer('last_error', torch.zeros(1))

    def forward(self, setpoint: torch.Tensor, current_value: torch.Tensor, dt: float = 0.1) -> torch.Tensor:
        """
        PID 제어 로직을 수행하여 제어 출력을 계산합니다.
        """
        error = setpoint - current_value

        # ### 수정된 부분 ###
        # 이전 integral_term의 계산 기록을 분리(detach)하여 그래프가 무한히 길어지는 것을 방지
        integral_update = self.integral_term.detach() + error * dt
        self.integral_term = integral_update

        # 미분항 계산
        derivative_term = (error - self.last_error) / dt

        # PID 제어 출력 계산
        output = (self.Kp * error) + \
                 (self.Ki * self.integral_term) + \
                 (self.Kd * derivative_term)

        # 다음 스텝을 위해 현재 오차를 그래디언트 흐름 없이 저장
        self.last_error = error.detach()

        return output

    def reset(self):
        """시뮬레이션 에피소드 시작 시 내부 상태를 초기화합니다."""
        self.integral_term.zero_()
        self.last_error.zero_()