# DEEP COFFEE - Computer Vision For Coffee Beans Selection
[![Build Status](https://travis-ci.com/Tauranis/deep_coffee.svg?branch=master)](https://travis-ci.com/Tauranis/deep_coffee)

![](https://media.giphy.com/media/OR0A0UC0YbYhZBQ9qS/giphy.gif)

## Motivation

Coffee is the second most consumed beverage in the world, behind only from water. It arrived in 1727 in Brazil, which became the largest producer and exporter of beans on the planet, as well as the second main consumer.
The country has consolidated itself as a business giant, where the 2018 harvest was 45 million bags(60Kg each), with only 46 % for home consumption.

In view of the above, automation in grain selection shows to be useful as a way to increase the added value and quality of the final product.
For example, a grain classified as gourmet by the Brazilian Coffee Industry Association (ABIC) may cost up to three times more than one classified as normal.

## Goal

The goal of this post is to evaluate convolutional neural networks for the task of coffee beans classification.
In addition, the database used in this work was built from scratch and is publicly available.
For reproducibility, the source code and database are available here for free.
Yet this is another repo out of a plethora regarding image classification (and transfer learning as well), the innovation doesn’t necessarily come from the technique itself, but the problem it solves.

## Database

The database used for this work was built from scratch.
Coffee beans were arranged on a flat white surface, separated from each other.

The capture device used was a camera of a Lenovo Vibe K5 mobile phone arranged 10cm from the beans. In addition, a LED flashlight was used to reduce any shadow effect caused by the natural sunlight.

For more complete info, read this [cool post](https://medium.com/swlh/automation-for-coffee-bean-selection-79a6b32b88de).

**Obs:** It was chosen not to use object detection algorithms because this is has very controlled environment case, with homogeneous background, high contrast between background & foreground, lightening conditions and non-overlapping objects. 
Although an object detection would work perfectly, it would be overkill.
I wanted to show that computer vision is not only about deep learning (althought I've used a CNN for classification :P).
We can solve simple problems with simple approaches.

## Installation

### Install docker

[Follow these steps](https://docs.docker.com/install/linux/docker-ce/ubuntu/)

### Install nvidia-container-toolkit or nvidia-docker2

[Follow these steps](https://github.com/NVIDIA/nvidia-docker)

### Install

The command below will download the dataset, build the docker image and run the data preprocessing steps (data augmentation and create tfrecords)

``` 
make install
```

### Build docker image

This is already performed if you previously have run `make install` 

``` 
docker build . -t deep_coffee
```

### Test if image was build correctly

``` 
docker run --rm --gpus all deep_coffee nvidia-smi
```

### Playground - enter inside container

``` 
docker run -it \
-v ${PWD}/deep_coffee:/src/deep_coffee \
-v ${PWD}/test:/src/test \
-v ${PWD}/dataset:/dataset \
-v ${PWD}/trained_models:/trained_models \
-v ${PWD}/keras_pretrained_models:/root/.keras/models/ \
-p 6006:6006 \
--rm --gpus all deep_coffee bash
```

#### Run unit tests

``` 
docker run \
--rm --gpus all deep_coffee \
python -m unittest discover -s /app/test/image_proc
```

#### Crop beans

This is already performed if you previously have run `make install` 

``` 
docker run \
-v ${PWD}/dataset:/dataset \
--rm --gpus all deep_coffee \
python -m deep_coffee.image_proc.crop_beans \
--raw_images_dir /dataset/raw \
--output_dir /dataset/cropped
```

#### Data Augmentation

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

#### Generate TFRecords

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

#### Decode dataset from tfrecords to images (debug)
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

#### Train network

**BEWARE** when training **ResNet** or **VGG** locally on your computer, you're likely to get OOM. Choose the batch size wisely.

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
--config_file /app/deep_coffee/ml/config/coffee_net_v1.yml \
--learning_rate 0.0001 \
--batch_size 8
```

#### Project Embeddings on TensorBoard

```
docker run \
-v ${PWD}/dataset:/dataset \
-v ${PWD}/trained_models:/trained_models \
-v ${PWD}/deep_coffee:/src/deep_coffee \
-v ${PWD}/keras_pretrained_models:/root/.keras/models/ \
--rm --gpus all deep_coffee \
python -m deep_coffee.ml.project_embeddings \
--tfrecord_path "/dataset/tfrecords/eval*" \
--output_dir /trained_models/coffee_net_v1/20200112-210616 \
--ckpt_path /trained_models/coffee_net_v1/20200112-210616/model.hdf5 \
--tft_artifacts_dir /dataset/tft_artifacts \
--layer_name head_dense_1 \
--dataset_len 264 \
--input_dim 224
```

#### Make prediction from a SavedModel

```
docker run \
-v ${PWD}/dataset:/dataset \
-v ${PWD}/trained_models:/trained_models \
-v ${PWD}/deep_coffee:/src/deep_coffee \
-v ${PWD}/keras_pretrained_models:/root/.keras/models/ \
--rm --gpus all deep_coffee \
python -m deep_coffee.ml.load_and_test_saved_model \
--model_path /trained_models/coffee_net_v1/20200112-230339/saved_model/ \
--sample_image_path /dataset/bad/1acf8679-d50f-446c-b806-def7d073e244_135.jpg
```
