import torch
import torch.nn as nn

class MultiDimPlant(nn.Module):
    """
    d_model 차원의 상태 벡터를 가진 가상의 다차원 플랜트.
    내부적으로는 서로 독립적인 d_model개의 1차 지연 시스템으로 구성됩니다.
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model

        # 각 차원별로 다른 특성을 갖도록 ku와 tau를 랜덤하게 생성
        # ku: 1.0 ~ 3.0, tau: 3.0 ~ 7.0
        kus = torch.rand(d_model) * 2.0 + 1.0
        taus = torch.rand(d_model) * 4.0 + 3.0

        self.ku = nn.Parameter(kus, requires_grad=False)
        self.tau = nn.Parameter(taus, requires_grad=False)
        
        self.register_buffer('state', torch.zeros(d_model))

    def step(self, control_inputs: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
        """
        주어진 d_model 차원의 제어 입력 벡터에 대해 시스템의 다음 상태를 계산합니다.

        Args:
            control_inputs (torch.Tensor): 제어기(GeometricPIDNet)로부터의 제어 입력 벡터.
            dt (torch.Tensor): 시뮬레이션 시간 간격.

        Returns:
            torch.Tensor: 업데이트된 시스템의 새로운 상태 벡터.
        """
        y = self.state
        dydt = (-y + self.ku * control_inputs) / self.tau
        new_state = y + dydt * dt
        
        self.state = new_state.detach()
        return new_state

    def reset(self):
        """시스템 상태를 0 벡터로 초기화합니다."""
        self.state.zero_()