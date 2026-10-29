# 基础镜像锁定 CUDA 12.1
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    python3-venv \
    ffmpeg \
    libvorbis-dev \
    libsndfile1 \
    sox \
    libsox-fmt-all \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip setuptools wheel

RUN python3 -m pip install --no-cache-dir \
    torch==2.3.1+cu121 \
    torchvision==0.18.1+cu121 \
    torchaudio==2.3.1+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

COPY requirements.txt /tmp/requirements.txt

RUN grep -vE '^(torch|torchvision|torchaudio)' /tmp/requirements.txt > /tmp/regs_no_torch.txt \
    && python3 -m pip install --no-cache-dir -r /tmp/regs_no_torch.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

RUN rm -f /tmp/regs_no_torch.txt

WORKDIR /app
COPY . .

RUN python3 -c "import torch, torchaudio; assert torch.version.cuda.startswith('12.1'), f'CUDA版本不匹配: {torch.version.cuda}'; print('[OK] torch', torch.__version__, 'cuda', torch.version.cuda)"

# 暴露端口
EXPOSE 18880

# CMD ["python3", "app.py"]
