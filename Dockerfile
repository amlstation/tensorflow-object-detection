FROM tensorflow/tensorflow:2.12.0

ARG DEBIAN_FRONTEND=noninteractive

RUN apt update

RUN apt install -y python3-pip

RUN python3 -m pip install --upgrade pip

RUN apt-get update && apt-get install -y \
    git \
    gpg-agent \
    python3-cairocffi \
    protobuf-compiler \
    python3-pil \
    python3-lxml \
    python3-tk \
    wget \
    libgl1 \
    unzip

COPY requirements.txt requirements.txt

RUN pip3 install -r requirements.txt

RUN mkdir -p /workspace

COPY ./tf_object_detection_api /workspace/tf_object_detection_api

RUN cd /workspace/tf_object_detection_api/research \
    && protoc object_detection/protos/*.proto --python_out=. \
    && cp object_detection/packages/tf2/setup.py . \
    && python3 -m pip install .

WORKDIR /workspace

ENV TF_CPP_MIN_LOG_LEVEL 2
