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

## Playground

Enter container
```
docker run -it -v ${PWD}/deep_coffee:/src/deep_coffee -v ${PWD}/test:/src/test --rm --gpus all deep_coffee bash
```


### Run unit tests
```
python -m unittest test.image_proc.test_OpenCVStream
```

