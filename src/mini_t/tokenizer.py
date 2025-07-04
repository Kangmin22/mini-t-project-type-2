# FILE: src/mini_t/tokenizer.py
from tokenizers import ByteLevelBPETokenizer
from transformers import PreTrainedTokenizerFast
from typing import List, Optional
import os

class MiniTTokenizer:
    """
    ByteLevelBPETokenizer를 훈련하고 Hugging Face의 PreTrainedTokenizerFast로
    래핑하여 호환성을 확보하는 클래스입니다.
    """
    def __init__(self, tokenizer_obj: Optional[PreTrainedTokenizerFast] = None):
        self.tokenizer = tokenizer_obj

    def train(self, files: List[str], vocab_size: int, min_frequency: int, special_tokens: List[str], save_path: str):
        """
        주어진 텍스트 파일로부터 토크나이저를 훈련합니다.

        Args:
            files (List[str]): 훈련에 사용할 텍스트 파일 경로 리스트.
            vocab_size (int): 목표 어휘 크기.
            min_frequency (int): 토큰이 되기 위한 최소 등장 빈도.
            special_tokens (List[str]): 특수 토큰 리스트.
            save_path (str): 훈련된 토크나이저를 저장할 경로.
        """
        # 1. ByteLevelBPETokenizer 초기화
        tokenizer = ByteLevelBPETokenizer()

        # 2. 토크나이저 훈련
        tokenizer.train(files=files, vocab_size=vocab_size, min_frequency=min_frequency, special_tokens=special_tokens)

        # 3. 훈련된 토크나이저를 PreTrainedTokenizerFast로 래핑
        # ByteLevelBPETokenizer는 vocab.json과 merges.txt를 생성합니다.
        # 이 파일들을 PreTrainedTokenizerFast가 읽어서 로드합니다.
        os.makedirs(save_path, exist_ok=True)
        tokenizer.save_model(save_path)
        
        # 저장된 파일로부터 PreTrainedTokenizerFast 객체 로드
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained(save_path)
        print(f"Tokenizer trained and saved to {save_path}")

    def encode(self, text: str) -> List[int]:
        """텍스트를 토큰 ID 리스트로 인코딩합니다."""
        if self.tokenizer is None:
            raise ValueError("Tokenizer is not trained or loaded.")
        return self.tokenizer.encode(text)

    def decode(self, token_ids: List[int]) -> str:
        """토큰 ID 리스트를 텍스트로 디코딩합니다."""
        if self.tokenizer is None:
            raise ValueError("Tokenizer is not trained or loaded.")
        return self.tokenizer.decode(token_ids)

    def save(self, save_path: str):
        """토크나이저를 저장합니다."""
        if self.tokenizer is None:
            raise ValueError("Tokenizer is not trained or loaded.")
        self.tokenizer.save_pretrained(save_path)

    @classmethod
    def from_pretrained(cls, load_path: str) -> 'MiniTTokenizer':
        """저장된 토크나이저를 로드합니다."""
        tokenizer_obj = PreTrainedTokenizerFast.from_pretrained(load_path)
        return cls(tokenizer_obj)