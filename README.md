# DEEP COFFEE - Deep Learning for coffee beans selection

## Install docker

[Follow these steps](https://docs.docker.com/install/linux/docker-ce/ubuntu/)

## Install nvidia-container-toolkit or nvidia-docker2

[Follow these steps](https://github.com/NVIDIA/nvidia-docker)

## Build image

```
docker build . -t deep_coffee
```

## Test image

```
docker run --rm --gpus all deep_coffee nvidia-smi
```