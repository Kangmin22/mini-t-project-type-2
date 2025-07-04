Project Mini-T-TYPE-2: 기하학적 제어 및 MLOps 파이프라인
**"Project Mini-T-TYPE-2"**는 "Transformer as Geometric Flow"라는 이론적 프레임워크를 실제 MLOps 파이프라인으로 구현하는 연구 개발 프로젝트입니다.

이 프로젝트는 Project Mini-T의 성과를 계승하여, 기존의 스칼라(Scalar) 기반 제어기를 고차원 벡터(Vector) 공간에서 작동하는 **기하학적 제어기(GeometricPIDNet)**로 진화시키는 것을 목표로 합니다. 최종 결과물은 이론 검증부터 모델 훈련까지의 전 과정을 자동화하고 완벽하게 재현할 수 있는 종단간(End-to-End) MLOps 파이프라인입니다.

핵심 이론: 기하학적 흐름 (Geometric Flow)
본 프로젝트는 LLM을 단순한 기호 처리 시스템이 아닌, 고도로 구조화된 의미 다양체(Semantic Manifold) 위에서 작동하는 **동역학 시스템(Dynamical System)**으로 바라보는 새로운 관점을 채택했습니다.

이 관점에서, 우리는 모델의 학습 과정을 단순히 Loss를 최소화하는 것이 아닌, 의미 다양체 위에서 이상적인 경로를 따라가는 '흐름'을 제어하는 문제로 재정의했습니다.

주요 특징 및 성과
GeometricLoss 개발: 모델 출력의 '방향성'을 제어하는 공명(Resonance) 항(alpha)을 도입하여, 결과뿐만 아니라 과정의 질을 평가하는 새로운 Loss 함수를 구현했습니다.

지능형 HPO: '흐름의 효율성'을 측정하는 발산(Divergence) 항(beta)을 HPO 목표에 추가했습니다. 실험 결과, alpha와 beta가 모두 유의미한 값으로 선택되어 우리 이론의 실효성을 정량적으로 입증했습니다.

GeometricPIDNet 구현: PID 제어기의 게인(Kp, Ki, Kd)을 학습 가능한 행렬(nn.Linear)로 구현하여, 고차원 벡터를 직접 제어하는 **진정한 의미의 '기하학적 제어기'**를 성공적으로 훈련시켰습니다.

종단간 재현성: Docker와 Conda를 통해 개발 및 실행 환경을 완벽하게 캡슐화했으며, click 기반의 마스터 스크립트(run_pipeline.py)를 통해 모든 실험 과정을 단일 명령어로 자동화하고 재현할 수 있습니다.

프로젝트 구조
mini-t-project-type-2/
├── scripts/
│   ├── check_geometric_pid.py
│   ├── run_phase1_training.py
│   ├── run_phase3_training.py
│   ├── run_pipeline.py
│   └── train_final_model.py
├── src/
│   ├── hpo/
│   │   └── optimize.py
│   ├── pid_training/
│   │   ├── geometric_pid_net.py
│   │   ├── multi_dim_plant.py
│   │   ├── pid_net.py
│   │   └── plant.py
│   └── utils/
│       └── loss.py
├── tests/
│   ├── test_geometric_pid_net.py
│   ├── test_mini_t.py
│   └── test_pid_net.py
├── Dockerfile
├── docker-compose.yml
├── environment.yml
├── pytest.ini
└── README.md
시작하기
전제 조건
Git

Docker Desktop

NVIDIA GPU 및 관련 드라이버 (로컬 또는 Colab)

1. 로컬 개발 환경 설정
Bash

# 1. 저장소 복제
git clone https://github.com/Kangmin22/mini-t-project-type-2.git
cd mini-t-project-type-2

# 2. Docker 컨테이너 빌드 및 실행
docker-compose up --build -d

# 3. 컨테이너 접속
docker-compose exec app bash

# 4. Conda 환경 활성화
conda activate mini-t
2. 자동화 파이프라인 실행
컨테이너 내부에서 마스터 스크립트를 사용하여 주요 작업을 실행할 수 있습니다.

Bash

# 모든 사용 가능한 명령어 확인
python -m scripts.run_pipeline --help

# 로컬에서 실행 가능한 전체 파이프라인 실행
# (Sanity Check -> Final PIDNet 훈련 -> GeometricPIDNet 훈련)
python -m scripts.run_pipeline run-all
3. HPO 실행 (Google Colab)
리소스 집약적인 HPO 작업은 Google Colab의 GPU 환경에서 실행하는 것을 권장합니다.

Python

# Colab 셀에서 실행

# 1. GitHub 저장소 복제 및 최신 코드로 업데이트
!git clone https://github.com/Kangmin22/mini-t-project-type-2.git
%cd mini-t-project-type-2

# 2. 필요 라이브러리 설치
!pip install -r requirements.txt

# 3. HPO 스크립트 실행
!python -m src.hpo.optimize
주요 실험 결과
최적 하이퍼파라미터: HPO를 통해 alpha ≈ 0.346, beta ≈ 0.086 이라는 최적의 가중치를 발견했으며, 이는 우리 이론의 두 핵심 요소가 모두 모델 성능 향상에 기여했음을 증명합니다.

최종 훈련 모델: 파이프라인 실행 시 다음의 두 가지 핵심 결과물이 생성됩니다.

final_pid_net.pth: 기하학적 Loss로 최적화된 표준 PID 제어기

geometric_pid_net.pth: 고차원 벡터를 직접 제어하는 기하학적 PID 제어기

향후 연구 방향
본 프로젝트의 성공은 다음과 같은 더 도전적인 후속 연구의 발판이 됩니다.

LLM과의 직접 통합: GeometricPIDNet을 실제 트랜스포머 모델의 레이어 사이에 적용하여 의미 흐름을 능동적으로 제어하는 연구.

PEFT와의 융합: QLoRA와 같은 미세조정 기법에 기하학적 제어 이론을 접목하여 더 안정적이고 효율적인 파인튜닝 프레임워크를 개발하는 연구.