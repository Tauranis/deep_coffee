# DEEP COFFEE - Deep Learning for coffee beans selection
[![Build Status](https://travis-ci.com/Tauranis/deep_coffee.svg?branch=master)](https://travis-ci.com/Tauranis/deep_coffee)

## Install docker

[Follow these steps](https://docs.docker.com/install/linux/docker-ce/ubuntu/)

## Install nvidia-container-toolkit or nvidia-docker2

[Follow these steps](https://github.com/NVIDIA/nvidia-docker)

## Download dataset
It won't be download the augmented data, but only the original images.
To perform data augmentation, read the full doc.

```
make install
```

## Build image

```
docker build . -t deep_coffee
```

## Test image

```
docker run --rm --gpus all deep_coffee nvidia-smi
```

## Playground
```
docker run -it \
-v ${PWD}/deep_coffee:/src/deep_coffee \
-v ${PWD}/test:/src/test \
-v ${PWD}/dataset:/dataset \
--rm --gpus all deep_coffee bash
```


### Run unit tests
```
docker run \
--rm --gpus all deep_coffee \
python -m unittest discover -s /app/test/image_proc
```

### Crop beans 
```
docker run \
-v ${PWD}/dataset:/dataset \
--rm --gpus all deep_coffee \
python -m deep_coffee.image_proc.crop_beans \
--raw_images_dir /dataset/raw \
--output_dir /dataset/cropped
```


### Data Augmentation

Up to this day, only rotation is implemented

**TODO**:

* Saturation & Brightness
* Noise
* GANs

#### Rotate beans 

Good beans
```
docker run \
-v ${PWD}/dataset:/dataset \
--rm --gpus all deep_coffee \
python -m deep_coffee.image_proc.data_aug \
--input_dir /dataset/good \
--output_dir /dataset/good \
--angle_list 45,90,135,180,225,270
```

Bad beans
```
docker run \
-v ${PWD}/dataset:/dataset \
--rm --gpus all deep_coffee \
python -m deep_coffee.image_proc.data_aug \
--input_dir /dataset/bad \
--output_dir /dataset/bad \
--angle_list 45,90,135,180,225,270
```