# Venus

## VLLM Deployment

### Environment Setup

```bash
# Create and activate conda environment
conda create -n flash python=3.10 -y
conda activate flash
# Install CUDA Toolkit and cuDNN (CUDA 12.4)
conda install -c nvidia/label/cuda-12.4.0 cuda-toolkit
conda install -c nvidia/label/cuda-12.4.0 cudnn
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.4 -c pytorch -c nvidia
# Install FlashAttention
wget https://github.com/Dao-AILab/flash-attention/releases/download/flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
pip install  flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
pip install transformers==4.40.0
pip install opencv-python-headless
pip install av
pip install 'accelerate>=0.26.0'
pip install decord
pip install easyocr
pip install faiss-gpu
pip install spacy
pip install ffmpeg-python
pip install ultralytics
```

**Note**: The version of FlashAttention should match your CUDA and PyTorch environment. Please refer to the [FlashAttention release page](https://github.com/Dao-AILab/flash-attention/releases) to download the appropriate `.whl` file for your setup.

### Docker Setup

```bash
distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
   && curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add - \
   && curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

#启动容器
sudo docker run -d --name cuda_124 \
--restart unless-stopped --gpus all   \
-p 5003:22 -p 5001:8082  \
--mount type=bind,source=/home/eco,target=/home   \
pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel tail -f /dev/null

# 进入容器
docker exec -it cuda_124 /bin/bash

# 安装 SSH 服务
apt update && apt install -y openssh-server
mkdir /var/run/sshd

# 允许 root 登录（仅限测试环境）
sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config

passwd root
ssh-keygen -A
service ssh restart
/usr/sbin/sshd -p 5003

vim ~/.bashrc
#添加如下内容
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libcuda.so.1

cd home
apt install libgl1-mesa-glx libglu1-mesa
bash Anaconda3-2024.06-1-Linux-x86_64.sh
conda create -n Venus python=3.10
# Install CUDA Toolkit and cuDNN (CUDA 12.4)
conda install -c nvidia/label/cuda-12.4.0 cuda-toolkit
conda install -c nvidia/label/cuda-12.4.0 cudnn
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.4 -c pytorch -c nvidia
# Install FlashAttention
wget https://github.com/Dao-AILab/flash-attention/releases/download/flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
pip install  flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
pip install transformers==4.40.0
pip install opencv-python-headless
pip install av
pip install 'accelerate>=0.26.0'
pip install decord
pip install easyocr
pip install faiss-gpu
pip install spacy
pip install ffmpeg-python
pip install ultralytics
pip install open_clip_torch
```

### Quick Start

- Checkpoint: [LLaVA-Video-7B-Qwen2](https://huggingface.co/lmms-lab/LLaVA-Video-7B-Qwen2)

- Demo Script: `video_demo.py` (runs VLM inference on a sample video)

```bash
bash scripts/run_llava_video.sh    # Run inference on the demo video
```

**Note**: You may need to adjust the "device_map" based on your available GPUs.

## Venus Deployment

- Demo Script: `venus_demo.py`

```bash
bash scripts/run_llava_video_venus.sh
```

### Evaluation

#### Video-MME Benchmark

- Dataset: https://huggingface.co/datasets/lmms-lab/Video-MME/tree/main
- Evaluation Pipeline: For detailed instructions on how to run the evaluation, please refer to the official guide:
  https://github.com/BradyFU/Video-MME?tab=readme-ov-file#-evaluation-pipeline

**Example** :
Testing LLaVA-Video-7B-Qwen2 on the VideoMME benchmark:

1. Run the generation script:

```bash
python eval/VideoMME/eval_llava_video_videomme.py  --video_duration_type medium --max_frames_num  32
```

2. The generated answers will be saved to: `eval/results/eval_llava_video_videomme_medium_32.json`

3. Evaluate the results:

```bash
python eval/VideoMME/eval_videomme.py --results_file eval/results/eval_llava_video_videomme_medium_32.json --video_duration_type medium

```
