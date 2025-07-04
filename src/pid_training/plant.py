import torch
import torch.nn as nn

class FirstOrderPlant(nn.Module):
    """
    학습 가능한 PIDNet을 훈련시키기 위한 가상의 1차 지연 시스템(FOPDT) 시뮬레이터.
    이전 프로젝트의 교훈을 반영하여 scipy.integrate.odeint 의존성을 제거하고,
    미분 가능한 순수 PyTorch 텐서 연산(오일러 방법)으로 재구현함.
    """
    def __init__(self, ku: float, tau: float, initial_state: float = 0.0):
        """
        Args:
            ku (float): 시스템 이득 (System Gain).
            tau (float): 시정수 (Time Constant).
            initial_state (float): 시스템의 초기 상태 값.
        """
        super().__init__()
        # 이 파라미터들은 외부에서 주어지며, 학습 대상이 아님
        self.ku = torch.tensor(ku, dtype=torch.float32)
        self.tau = torch.tensor(tau, dtype=torch.float32)
        
        # register_buffer를 사용하여 state를 모델의 상태로 등록하되, 학습 파라미터에서는 제외
        self.register_buffer('state', torch.tensor(initial_state, dtype=torch.float32))

    def step(self, control_input: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
        """
        주어진 제어 입력(u)과 시간 간격(dt)에 대해 시스템의 다음 상태를 계산합니다.
        미분 방정식: dy/dt = (-y + Ku * u) / tau
        오일러 1차 근사: y_next = y + dt * dy/dt

        Args:
            control_input (torch.Tensor): 제어기(PIDNet)로부터의 제어 입력 (u).
            dt (torch.Tensor): 시뮬레이션 시간 간격 (time step).

        Returns:
            torch.Tensor: 업데이트된 시스템의 새로운 상태 값.
        """
        # 현재 상태 y(t)
        y = self.state
        
        # 미분 계수 dy/dt 계산
        dydt = (-y + self.ku * control_input) / self.tau
        
        # 다음 상태 y(t+dt) 계산 (오일러 방법)
        new_state = y + dydt * dt
        
        # 내부 상태 업데이트
        self.state = new_state.detach() # 다음 step을 위해 그래디언트 흐름을 끊음

        return new_state

    def reset(self, initial_state: float = 0.0):
        """
        시뮬레이션 에피소드 시작 시 시스템 상태를 초기화합니다.
        """
        self.state = torch.tensor(initial_state, dtype=torch.float32)