#FROM python:3.8-slim as builder
FROM nvidia/cuda:11.6.0-devel-ubuntu20.04
ENV PYTHONUNBUFFERED 1

RUN apt-get update -y

RUN DEBIAN_FRONTEND=noninteractive apt-get install -y ffmpeg libsm6 libxext6 libgl1
RUN DEBIAN_FRONTEND=noninteractive apt -y update
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y python3.10
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y pip

RUN pip install asgiref==3.5.2
RUN pip install autopep8==1.6.0
RUN pip install certifi==2022.6.15
RUN pip install charset-normalizer==2.1.0
RUN pip install idna==3.3
RUN pip install numpy==1.23.1
RUN pip install opencv-python==4.6.0.66
RUN pip install Pillow==9.2.0
RUN pip install pycodestyle==2.9.0
RUN pip install pytz==2022.1
RUN pip install PyYAML==6.0
RUN pip install requests==2.28.1
RUN pip install sqlparse==0.4.2
RUN pip install toml==0.10.2
# This installs the latest nightly build of PyTorch (its needed for cuda 11.6 support) - you can change this to a specific version if your gpu has a different cuda version
RUN pip3 install --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cu116
RUN pip install typing_extensions==4.3.0
RUN pip install tzdata==2022.1
RUN pip install urllib3==1.26.11
RUN pip install gunicorn


#install git
RUN apt install -y libnss3-tools git curl

# install additional libs here
RUN pip install albumentations 
RUN pip install tqdm 
RUN pip install matplotlib

RUN mkdir /backend

WORKDIR /backend

COPY . /backend/


EXPOSE 8000

