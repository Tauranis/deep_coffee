FROM tensorflow/tensorflow:2.0.0-gpu-py3
#FROM tensorflow/tensorflow:2.0.0-py3


# General dependencies
RUN apt-get update && apt-get install -y

RUN apt install -y  libsm6 libxext6 libxrender-dev

WORKDIR /app

COPY requirements.txt /assets/requirements.txt
RUN pip install -r /assets/requirements.txt

COPY deep_coffee /app/deep_coffee
COPY test /app/test
