import torch
import torch.nn as nn
import torch.nn.functional as F

class GeometricLoss(nn.Module):
    """
    "Transformer as Geometric Flow" 이론에 기반한 Loss 함수.
    - Resonance Loss: 의미 흐름의 '방향성'을 제어.
    - Curvature Loss: 의미 흐름의 '부드러움'을 제어.
    """
    def __init__(self, alpha: float, gamma: float):
        """
        Loss 함수의 가중치를 초기화합니다.

        Args:
            alpha (float): Resonance Loss의 가중치.
            gamma (float): Curvature Regularization Loss의 가중치.
        """
        super().__init__()
        if not (0.0 <= alpha <= 1.0):
            raise ValueError(f"alpha는 0과 1 사이의 값이어야 합니다. 현재 값: {alpha}")
        if not (gamma >= 0.0):
            raise ValueError(f"gamma는 0 이상의 값이어야 합니다. 현재 값: {gamma}")

        self.alpha = alpha
        self.gamma = gamma

    def calculate_resonance_loss(self, flow_vectors: torch.Tensor, target_vectors: torch.Tensor) -> torch.Tensor:
        """
        의미 흐름의 최종 벡터와 목표 임베딩 벡터 간의 코사인 유사도를 기반으로
        '공명 손실'을 계산합니다.
        """
        resonance_loss = -F.cosine_similarity(flow_vectors, target_vectors, dim=-1).mean()
        return resonance_loss

    def calculate_curvature_loss(self, x_curr: torch.Tensor, x_prev: torch.Tensor, x_next: torch.Tensor) -> torch.Tensor:
        """
        연속된 세 시점의 벡터를 사용하여 유한 차분법으로 2차 미분을 근사하고,
        '곡률 정칙화' 손실을 계산합니다.
        """
        second_derivative_approx = x_next + x_prev - 2 * x_curr
        curvature_regularization_loss = torch.norm(second_derivative_approx)
        return curvature_regularization_loss

    def forward(self, main_loss: torch.Tensor, model_outputs=None, target_vectors=None) -> torch.Tensor:
        """
        주어진 손실에 기하학적 손실 항들을 추가하여 최종 손실을 계산합니다.
        """
        final_loss = main_loss

        # Resonance Loss 계산
        if self.alpha > 0 and model_outputs is not None and target_vectors is not None:
            # ### 수정된 부분 ###
            # 딕셔너리의 키로 접근하도록 변경: model_outputs.hidden_states -> model_outputs['hidden_states']
            flow_vectors = model_outputs['hidden_states'][-1]
            resonance_loss = self.calculate_resonance_loss(flow_vectors, target_vectors)
            final_loss += self.alpha * resonance_loss

        return final_loss