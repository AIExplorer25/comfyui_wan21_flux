ARG CUDA_VERSION="11.8.0"
ARG CUDNN_VERSION="8"
ARG UBUNTU_VERSION="22.04"
ARG DOCKER_FROM=nvidia/cuda:$CUDA_VERSION-cudnn$CUDNN_VERSION-devel-ubuntu$UBUNTU_VERSION

# Base NVidia CUDA Ubuntu image
FROM $DOCKER_FROM AS base

# Install Python plus openssh, which is our minimum set of required packages.
RUN apt-get update -y && \
    apt-get install -y python3 python3-pip python3-venv && \
    apt-get install -y --no-install-recommends openssh-server openssh-client git git-lfs wget vim zip unzip curl && \
    python3 -m pip install --upgrade pip && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install nginx
RUN apt-get update && \
    apt-get install -y nginx

# Copy the 'default' configuration file to the appropriate location
COPY default /etc/nginx/sites-available/default

ENV PATH="/usr/local/cuda/bin:${PATH}"

# Install pytorch
ARG PYTORCH="2.5.1"
ARG CUDA="118"
RUN pip3 install --no-cache-dir -U torch==$PYTORCH torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu$CUDA

COPY --chmod=755 start-ssh-only.sh /start.sh
COPY --chmod=755 start-original.sh /start-original.sh
COPY --chmod=755 comfyui-on-workspace.sh /comfyui-on-workspace.sh
COPY --chmod=755 check_files.sh /check_files.sh

# Clone the git repo and install requirements in the same RUN command to ensure they are in the same layer
RUN git clone https://github.com/comfyanonymous/ComfyUI.git && \
    cd ComfyUI && \
    pip3 install -r requirements.txt && \
    cd custom_nodes && \
    git clone https://github.com/ltdrdata/ComfyUI-Manager.git && \
    git clone https://github.com/pythongosssss/ComfyUI-Custom-Scripts.git && \
    cd /ComfyUI && \
    mkdir pysssss-workflows

COPY --chmod=644 workflows/ /ComfyUI/user/default/workflows/
COPY --chmod=644 comfy.settings.json /ComfyUI/user/default/comfy.settings.json
COPY --chmod=644 example_photo.png /ComfyUI/input/example_photo.png
COPY --chmod=644 example_photo_small.png /ComfyUI/input/example_photo_small.png
COPY --chmod=644 example_pose.png /ComfyUI/input/example_pose.png
COPY --chmod=644 example2.png /ComfyUI/input/example2.png

WORKDIR /workspace

EXPOSE 8188

# Download and move flux_dev_example.png
RUN wget "https://github.com/comfyanonymous/ComfyUI_examples/blob/master/flux/flux_dev_example.png" -P /ComfyUI

# Install Xlabs-AI/flux-RealismLora
RUN apt-get install -y libgl1-mesa-glx libglib2.0-0


# This is a hacky way to change the default workflow on startup, but it works
COPY --chmod=644 defaultGraph.json /defaultGraph.json
COPY --chmod=755 replaceDefaultGraph.py /replaceDefaultGraph.py
# Run the Python script
RUN python3 /replaceDefaultGraph.py

# Overwrite the default.json file in ComfyUI/web/templates for the new UI
COPY --chmod=644 defaultGraph.json /ComfyUI/web/templates/default.json

# Add Jupyter Notebook
RUN pip3 install jupyterlab
EXPOSE 8888

# Add some additional custom nodes
# LDSR Upscale
RUN cd /ComfyUI/custom_nodes && \
    git clone https://github.com/flowtyone/ComfyUI-Flowty-LDSR.git && \
    cd ComfyUI-Flowty-LDSR && \
    pip3 install -r requirements.txt

# Add download scripts for additional models

# KJNodes
RUN cd /ComfyUI/custom_nodes && \
    git clone https://github.com/kijai/ComfyUI-KJNodes.git && \
    cd ComfyUI-KJNodes && \
    pip3 install -r requirements.txt

# rgthree
RUN cd /ComfyUI/custom_nodes && \
    git clone https://github.com/rgthree/rgthree-comfy.git && \
    cd rgthree-comfy && \
    pip3 install -r requirements.txt

# JPS-Nodes
RUN cd /ComfyUI/custom_nodes && \
    git clone https://github.com/JPS-GER/ComfyUI_JPS-Nodes.git

# Comfyrol Studio
RUN cd /ComfyUI/custom_nodes && \
    git clone https://github.com/Suzie1/ComfyUI_Comfyroll_CustomNodes.git

# ComfyUI-VideoHelperSuite
RUN cd /ComfyUI/custom_nodes && \
    git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git && \
    cd ComfyUI-VideoHelperSuite && \
    pip3 install -r requirements.txt


# ComfyUI-Impact-Subpack
RUN cd /ComfyUI/custom_nodes && \
    git clone https://github.com/ltdrdata/ComfyUI-Impact-Pack.git && \
    cd ComfyUI-Impact-Pack && \
    pip3 install -r requirements.txt && \
    python3 install.py

# ComfyUI-Impact-Subpack
RUN cd /ComfyUI/custom_nodes && \
    git clone https://github.com/ltdrdata/ComfyUI-Impact-Subpack && \
    cd ComfyUI-Impact-Subpack && \
    pip3 install -r requirements.txt

# make directory and download face_yolov8m.pt
RUN mkdir -p /ComfyUI/models/ultralytics/bbox && \
    wget "https://huggingface.co/Bingsu/adetailer/resolve/main/face_yolov8m.pt?download=true" -O /ComfyUI/models/ultralytics/bbox/face_yolov8m.pt

# ComfyUI-Impact-controlnet_aux
RUN cd /ComfyUI/custom_nodes && \
    git clone https://github.com/Fannovel16/comfyui_controlnet_aux.git && \
    cd comfyui_controlnet_aux && \
    pip3 install -r requirements.txt

# ComfyUI-UltimateSDUpscale
RUN cd /ComfyUI/custom_nodes && \
    git clone https://github.com/ssitu/ComfyUI_UltimateSDUpscale --recursive

# ComfyUI-Easy-Use
RUN cd /ComfyUI/custom_nodes && \
    git clone https://github.com/yolain/ComfyUI-Easy-Use.git && \
    cd ComfyUI-Easy-Use && \
    pip3 install -r requirements.txt

# was-node-suite-comfyui
RUN cd /ComfyUI/custom_nodes && \
    git clone https://github.com/WASasquatch/was-node-suite-comfyui.git && \
    cd was-node-suite-comfyui && \
    pip3 install -r requirements.txt

# ComfyUI-Logic
RUN cd /ComfyUI/custom_nodes && \
    git clone https://github.com/theUpsider/ComfyUI-Logic.git

# ComfyUI_essentials
RUN cd /ComfyUI/custom_nodes && \
    git clone https://github.com/cubiq/ComfyUI_essentials.git && \
    cd ComfyUI_essentials && \
    pip3 install -r requirements.txt

# cg-image-picker
RUN cd /ComfyUI/custom_nodes && \
    git clone https://github.com/chrisgoringe/cg-image-picker.git

# ComfyUI_LayerStyle
RUN cd /ComfyUI/custom_nodes && \
    git clone https://github.com/chflame163/ComfyUI_LayerStyle.git && \
    cd ComfyUI_LayerStyle && \
    pip3 install -r requirements.txt

# Ccomfyui-mixlab-nodes
RUN cd /ComfyUI/custom_nodes && \
    git clone https://github.com/shadowcz007/comfyui-mixlab-nodes.git && \
    cd comfyui-mixlab-nodes && \
    pip3 install -r requirements.txt

# comfyui-reactor-node
RUN cd /ComfyUI/custom_nodes && \
    git clone https://codeberg.org/Gourieff/comfyui-reactor-node.git && \
    cd comfyui-reactor-node && \
    pip3 install -r requirements.txt

# make the directory and download the swap_model file
RUN mkdir -p /ComfyUI/models/insightface && \
    wget "https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx?download=true" -O /ComfyUI/models/insightface/inswapper_128.onnx

# cg-use-everywhere
RUN cd /ComfyUI/custom_nodes && \
    git clone https://github.com/chrisgoringe/cg-use-everywhere.git

# ComfyUI-CogVideoXWrapper
RUN cd /ComfyUI/custom_nodes && \
    git clone https://github.com/kijai/ComfyUI-CogVideoXWrapper.git && \
    cd ComfyUI-CogVideoXWrapper && \
    pip3 install -r requirements.txt
	
# ComfyUI-CogVideoXWrapper
RUN cd /ComfyUI/custom_nodes && \
    git clone https://github.com/kijai/ComfyUI-WanVideoWrapper.git && \
    cd ComfyUI-WanVideoWrapper && \
    pip3 install -r requirements.txt	


# make the directory and download the model needed for the tutorial workflow on first launch
RUN mkdir -p /ComfyUI/models/checkpoints && \
    wget "https://huggingface.co/Comfy-Org/stable-diffusion-v1-5-archive/resolve/main/v1-5-pruned-emaonly-fp16.safetensors?download=true" -O /ComfyUI/models/checkpoints/v1-5-pruned-emaonly-fp16.safetensors


# Download and move clip_l.safetensors
RUN wget -O /ComfyUI/models/clip/clip_l.safetensors "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/clip_l.safetensors?download=true" --progress=bar:force:noscroll

# Download and move t5xxl_fp8_e4m3fn.safetensors
RUN wget -O /ComfyUI/models/clip/t5xxl_fp8_e4m3fn.safetensors "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp8_e4m3fn.safetensors?download=true" --progress=bar:force:noscroll

# Download LoRas
RUN wget -O /ComfyUI/models/loras/GracePenelopeTargaryenV5.safetensors "https://huggingface.co/WouterGlorieux/GracePenelopeTargaryenV5/resolve/main/GracePenelopeTargaryenV5.safetensors?download=true" --progress=bar:force:noscroll
RUN wget -O /ComfyUI/models/loras/VideoAditor_flux_realism_lora.safetensors "https://huggingface.co/VideoAditor/Flux-Lora-Realism/resolve/main/flux_realism_lora.safetensors?download=true" --progress=bar:force:noscroll

RUN pip3 install ffmpeg-python
RUN pip3 install git

RUN pip3  install huggingface_hub[hf_transfer]
RUN pip3  install hf_transfer


hf_hub_download(repo_id="Kijai/WanVideo_comfy", filename="Wan2_1-T2V-14B_fp8_e4m3fn.safetensors", local_dir="/ComfyUI/models/unet/")

hf_hub_download(repo_id="Kijai/WanVideo_comfy", filename="Wan2_1-I2V-14B-720P_fp8_e4m3fn.safetensors", local_dir="/ComfyUI/models/unet/")


hf_hub_download(repo_id="Kijai/WanVideo_comfy", filename="umt5-xxl-enc-fp8_e4m3fn.safetensors", local_dir="/ComfyUI/models/clip/")

hf_hub_download(repo_id="Kijai/WanVideo_comfy", filename="Wan2_1_VAE_fp32.safetensors", local_dir="/ComfyUI/models/vae/")




hf_hub_download(repo_id="Kijai/WanVideo_comfy", filename="open-clip-xlm-roberta-large-vit-huge-14_visual_fp16.safetensors", local_dir="/ComfyUI/models/clip/")

hf_hub_download(repo_id="Kijai/WanVideo_comfy", filename="umt5-xxl-enc-bf16.safetensors", local_dir="/ComfyUI/models/clip/")
 

hf_hub_download(repo_id="gemasai/4x_NMKD-Siax_200k", filename="4x_NMKD-Siax_200k.pth", local_dir="/ComfyUI/models/upscale_models/")

hf_hub_download(repo_id="ai-forever/Real-ESRGAN", filename="RealESRGAN_x8.pth", local_dir="/ComfyUI/models/upscale_models/")
 

CMD [ "/start.sh" ]
