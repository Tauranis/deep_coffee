FROM tensorflow/tensorflow:2.0.0-gpu-py3


# General dependencies
RUN apt-get update && apt-get install -y

WORKDIR /app

COPY requirements.txt /assets/requirements.txt
RUN pip install -r /assets/requirements.txt

COPY deep_coffee /app
COPY test /test
