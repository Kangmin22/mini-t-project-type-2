# FILE: Dockerfile (PyTorch 1.13.1 안정 버전 기준)
# 베이스 이미지를 PyTorch 1.13.1과 호환되는 CUDA 11.7.1로 변경합니다.
FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y wget git && rm -rf /var/lib/apt/lists/*

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh

ENV PATH /opt/conda/bin:$PATH
WORKDIR /app
ENV PYTHONPATH /app

# 최소한의 Conda 환경 yml 파일을 복사
COPY environment.yml .
# Conda 환경 생성 (Python 3.8 + pip만 설치됨)
RUN conda env create -f environment.yml

# Pip 요구사항 파일을 복사
COPY requirements.txt .
# Conda 환경 안에서 pip으로 모든 라이브러리 설치
RUN conda run -n mini-t pip install -r requirements.txt

# 대화형 셸(interactive shell)이 conda activate를 인식하도록 초기화
RUN conda init bash

# 기본 셸을 mini-t 환경으로 설정
SHELL ["conda", "run", "-n", "mini-t", "/bin/bash", "-c"]

CMD ["tail", "-f", "/dev/null"]