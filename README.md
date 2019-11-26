# DEEP COFFEE - Deep Learning for coffee beans selection
[![Build Status](https://travis-ci.com/Tauranis/deep_coffee.svg?branch=master)](https://travis-ci.com/Tauranis/deep_coffee)

## Install docker

[Follow these steps](https://docs.docker.com/install/linux/docker-ce/ubuntu/)

## Install nvidia-container-toolkit or nvidia-docker2

[Follow these steps](https://github.com/NVIDIA/nvidia-docker)

## Install

The command below will download the dataset, build the docker image and run the data preprocessing steps (data augmentation and create tfrecords)

``` 
make install
```

## Build docker image

This is already performed if you previously have run `make install` 

``` 
docker build . -t deep_coffee
```

## Test if image was build correctly

``` 
docker run --rm --gpus all deep_coffee nvidia-smi
```

## Playground - enter inside container

``` 
docker run -it \
-v ${PWD}/deep_coffee:/src/deep_coffee \
-v ${PWD}/test:/src/test \
-v ${PWD}/dataset:/dataset \
-v ${PWD}/trained_models:/trained_models \
-v ${PWD}/keras_pretrained_models:/root/.keras/models/ \
--rm --gpus all deep_coffee bash
```

### Run unit tests

``` 
docker run \
--rm --gpus all deep_coffee \
python -m unittest discover -s /app/test/image_proc
```

### Crop beans

This is already performed if you previously have run `make install` 

``` 
docker run \
-v ${PWD}/dataset:/dataset \
--rm --gpus all deep_coffee \
python -m deep_coffee.image_proc.crop_beans \
--raw_images_dir /dataset/raw \
--output_dir /dataset/cropped
```

### Data Augmentation

This is already performed if you previously have run `make install` 

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

### Generate TFRecords

This is already performed if you previously have run `make install` 

``` 
docker run \
-v ${PWD}/dataset:/dataset \
--rm --gpus all deep_coffee \
python -m deep_coffee.ml.images_to_tfrecords \
--output_dir /dataset/tfrecords \
--tft_artifacts_dir /dataset/tft_artifacts \
--good_beans_dir /dataset/good \
--good_beans_list_train /dataset/protocol/good_train.txt \
--good_beans_list_eval /dataset/protocol/good_eval.txt \
--good_beans_list_test /dataset/protocol/good_test.txt \
--bad_beans_dir /dataset/bad \
--bad_beans_list_train /dataset/protocol/bad_train.txt \
--bad_beans_list_eval /dataset/protocol/bad_eval.txt \
--bad_beans_list_test /dataset/protocol/bad_test.txt \
--image_dim 224 \
--n_shards 10 \
--ext jpg \
--temp-dir /tmp
```

### Decode dataset from tfrecords to images (just for testing)
```
docker run \
-v ${PWD}/dataset:/dataset \
-v ${PWD}/trained_models:/trained_models \
-v ${PWD}/deep_coffee:/src/deep_coffee \
-v ${PWD}/keras_pretrained_models:/root/.keras/models/ \
--rm --gpus all deep_coffee \
python -m deep_coffee.ml.decode_tfrecord_dataset \
--tfrecord_file "/dataset/tfrecords/train*" \
--output_dir /dataset/decoded_tfrecords \
--tft_artifacts_dir /dataset/tft_artifacts
```

### Train network

**BEWARE** when training **ResNet** or **VGG** locally on your laptop, you're likely to get OOM. Choose the batch size wisely.

``` 
export KERAS_HOME=/trained_models

docker run \
-v ${PWD}/dataset:/dataset \
-v ${PWD}/trained_models:/trained_models \
-v ${PWD}/deep_coffee:/src/deep_coffee \
-v ${PWD}/keras_pretrained_models:/root/.keras/models/ \
--rm --gpus all deep_coffee \
python -m deep_coffee.ml.train_and_evaluate \
--output_dir /trained_models \
--tft_artifacts_dir /dataset/tft_artifacts \
--input_dim 224 \
--trainset_len 1265 \
--evalset_len 264 \
--testset_len 278 \
--config_file /app/deep_coffee/ml/config/mobilenet.yml \
--learning_rate 0.0001 \
--batch_size 8
```

