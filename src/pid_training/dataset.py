# FILE: src/pid_training/dataset.py
import torch
from torch.utils.data import Dataset
import numpy as np

class PIDDataset(Dataset):
    """
    정적인 데이터를 로드하는 대신,
    매번 다른 시뮬레이션 설정을 동적으로 생성하는 데이터셋.
    """
    def __init__(self, num_samples: int, steps_per_episode: int):
        self.num_samples = num_samples
        self.steps_per_episode = steps_per_episode

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> dict:
        """
        하나의 시뮬레이션 에피소드에 대한 설정을 생성하여 반환합니다.
        """
        # 목표 값 시퀀스 생성 (예: 랜덤 계단 함수)
        num_changes = np.random.randint(1, 5)
        change_indices = np.sort(np.random.randint(1, self.steps_per_episode, num_changes))
        
        targets = torch.zeros(self.steps_per_episode)
        current_target = 0
        
        last_idx = 0
        for change_idx in change_indices:
            targets[last_idx:change_idx] = current_target
            current_target = np.random.uniform(-10, 10)
            last_idx = change_idx
        targets[last_idx:] = current_target

        # 플랜트 파라미터 랜덤 샘플링 (강건한 제어기 학습을 위해)
        plant_params = {
            "ku": np.random.uniform(1.0, 5.0),
            "tau": np.random.uniform(2.0, 10.0),
        }
        
        return {
            "target_setpoints": targets,
            "plant_params": plant_params,
        }