import torch
import ray
import transformers
import peft
import datasets
import bitsandbytes

print("="*50)
print("✅ 라이브러리 버전 검증 (최신 안정화 스택)")
print("="*50)
print(f"PyTorch version: {torch.__version__}")
print(f"Ray version: {ray.__version__}")
print(f"Transformers version: {transformers.__version__}")
print(f"PEFT version: {peft.__version__}")
print(f"Datasets version: {datasets.__version__}")
# bitsandbytes는 __version__ 속성이 없을 수 있으므로 예외 처리
try:
    print(f"BitsandBytes version: {bitsandbytes.__version__}")
except AttributeError:
    print("BitsandBytes version: (imported successfully, no __version__ attr)")
print("\n모든 핵심 라이브러리가 성공적으로 임포트되었습니다.")
print("="*50)