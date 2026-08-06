# NVIDIA NGC PyTorch（CUDA）；可按环境替换 tag
FROM nvcr.io/nvidia/pytorch:26.03-py3

# ==================== 1. 自动配置 pip 源 ====================
# 系统级配置（对root和非root用户都生效）
COPY pip.conf /etc/pip.conf
COPY requirements.txt /root/workspace/requirements.txt
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential git \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r /root/workspace/requirements.txt

# ==================== 2. 在这里安装需要的包 ====================
# 推荐写法：一次 RUN 安装多个包，节省镜像层数
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir jupyter jupyterlab

# 如果还想一次性装其他常用包，可以继续加在后面：
#     numpy pandas matplotlib scipy scikit-learn torch torchvision torchaudio \
#     && echo "Packages installed successfully"
# ========================================================

# （可选）验证安装是否成功
# RUN jupyter --version

ENV PYTHONPATH=/root/workspace/src

WORKDIR /root/workspace

CMD ["bash"]
