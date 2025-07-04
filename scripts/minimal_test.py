# FILE: scripts/minimal_test.py
import torch
import torch.optim as optim
import torch.multiprocessing as mp
import ray
from ray import tune
import os

def minimal_training_function(config):
    """Ray Tune으로 실행할 가장 단순한 PyTorch 학습 함수"""
    model = torch.nn.Linear(1, 1)
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])
    
    # 간단한 학습 루프
    for _ in range(5):
        x = torch.tensor([[1.0]])
        y = model(x)
        loss = (y - 1.0) ** 2
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    # 결과 리포트
    tune.report(loss=loss.item())
    return {"loss": loss.item()}

def run_minimal_test():
    """최소 기능 테스트를 실행합니다."""
    try:
        # local_mode=True로 단일 프로세스에서 안전하게 실행
        ray.init(local_mode=True, ignore_reinit_error=True)

        search_space = {"lr": tune.loguniform(1e-4, 1e-1)}
        
        tuner = tune.Tuner(
            minimal_training_function,
            param_space=search_space,
            tune_config=tune.TuneConfig(
                num_samples=1 # 단 하나의 샘플만 테스트
            ),
        )

        print("="*50)
        print("최소 기능 충돌 테스트를 시작합니다...")
        print("="*50)
        results = tuner.fit()
        
        best_result = results.get_best_result(metric="loss", mode="min")
        print("\n=======================================================")
        print("✅ 최소 기능 테스트 성공!")
        print(f"  - 최종 손실: {best_result.metrics['loss']:.4f}")
        print("  - 사용된 lr: {best_result.config['lr']:.4f}")
        print("=======================================================")

    finally:
        ray.shutdown()

if __name__ == "__main__":
    # Ray+PyTorch 충돌을 막기 위한 모든 안전장치를 적용합니다.
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    run_minimal_test()