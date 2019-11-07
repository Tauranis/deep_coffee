FROM tensorflow/tensorflow:2.0.0-gpu-py3
# FROM tensorflow/tensorflow:2.0.0-py3

# General dependencies
RUN apt-get update && apt-get install -y

RUN apt install -y  libsm6 libxext6 libxrender-dev

WORKDIR /app

RUN apt install wget \
    && mkdir /assets \
    && wget https://files.pythonhosted.org/packages/37/71/ebe308ba37bd2d4d56c436fb5baa846260d74ef1d6392d370bd1adee424e/tensorflow_gpu-2.0.0rc2-cp36-cp36m-manylinux2010_x86_64.whl -O /assets/tensorflow_gpu-2.0.0rc2-cp36-cp36m-manylinux2010_x86_64.whl

COPY requirements.txt /assets/requirements.txt
RUN pip install -r /assets/requirements.txt \
    && pip install /assets/tensorflow_gpu-2.0.0rc2-cp36-cp36m-manylinux2010_x86_64.whl --upgrade

COPY deep_coffee /app/deep_coffee
COPY test /app/test
