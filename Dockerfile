FROM nvidia/cuda:11.2.2-cudnn8-devel
#sudo docker run --gpus all nvidia/cuda:11.2.2-cudnn8-devel nvidia-smi

RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys A4B469963BF863CC && \
    apt-get update && \
    apt-get install -y curl vim && \
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add - && \
    echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | tee /etc/apt/sources.list.d/coral-edgetpu.list && \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get install -y python3-pip python3-dev python-is-python3 ffmpeg libsm6 libxext6 git edgetpu-compiler && \
    python3 -m pip install --upgrade pip && \
    pip install --no-cache-dir numpy torch>=1.7 opencv_python loguru tqdm torchvision thop ninja tabulate psutil tensorboard pycocotools>=2.0.2 onnx==1.8.1 onnxruntime==1.8.0 onnx-simplifier==0.3.5 torch_tb_profiler jupyter glob2 openvino==2021.4.2 openvino-dev==2021.4.2 tensorflow gdown tflite_runtime==2.11.0 tensorflow-datasets && \
    rm -rf /var/lib/apt/lists/* && \
    rm -rf /tmp/*

WORKDIR /data/catflap
COPY . /data/catflap
WORKDIR /data/catflap/yolox

RUN pip install --no-cache-dir -r requirements.txt
RUN pip install -e .

ENV TF_ENABLE_ONEDNN_OPTS=0

