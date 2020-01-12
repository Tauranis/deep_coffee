import pandas as pd
import numpy as np
import cv2
from tqdm import tqdm

import argparse
import os

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorboard.plugins import projector

from tensorflow_transform import TFTransformOutput
from tensorflow_transform.beam.tft_beam_io import transform_fn_io

from deep_coffee.ml.utils import list_tfrecords
from deep_coffee.ml.train_and_evaluate import input_fn
from deep_coffee.ml.models import model_zoo, preproc_zoo

import logging
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def images_to_sprite(data):
    """Creates the sprite image along with any necessary padding
    Args:
        data: NxHxW[x3] tensor containing the images.
    Returns:
        data: Properly shaped HxWx3 image with any necessary padding.
    Source: https://github.com/tensorflow/tensorflow/issues/6322
    """
    if len(data.shape) == 3:
        data = np.tile(data[..., np.newaxis], (1, 1, 1, 3))
    data = data.astype(np.float32)
    min = np.min(data.reshape((data.shape[0], -1)), axis=1)
    data = (data.transpose(1, 2, 3, 0) - min).transpose(3, 0, 1, 2)
    max = np.max(data.reshape((data.shape[0], -1)), axis=1)
    data = (data.transpose(1, 2, 3, 0) / max).transpose(3, 0, 1, 2)
    
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = ((0, n ** 2 - data.shape[0]), (0, 0),
               (0, 0)) + ((0, 0),) * (data.ndim - 3)
    data = np.pad(data, padding, mode='constant',
                  constant_values=0)
    # Tile the individual thumbnails into an image.
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3)
                                                           + tuple(range(4, data.ndim + 1)))
    data = data.reshape(
        (n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    data = (data * 255).astype(np.uint8)
    data = cv2.cvtColor(data,cv2.COLOR_BGR2RGB)
    return data


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--tfrecord_path", required=True)
    parser.add_argument("--tft_artifacts_dir", required=True)
    parser.add_argument("--input_dim", required=False, default=224, type=int)
    parser.add_argument("--dataset_len", required=True, default=264, type=int)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--ckpt_path", required=True)
    parser.add_argument("--layer_name", required=True)

    args = parser.parse_args()

    # Load dataset
    tfrecords_list = list_tfrecords(args.tfrecord_path)
    input_shape = [args.input_dim, args.input_dim, 3]

    tft_metadata_dir = os.path.join(
        args.tft_artifacts_dir, transform_fn_io.TRANSFORM_FN_DIR)
    tft_metadata = TFTransformOutput(args.tft_artifacts_dir)
    preproc_fn = preproc_zoo.get_preproc_fn("coffee_net_v1")

    input_data = input_fn(tfrecords_path=tfrecords_list,
                          tft_metadata=tft_metadata,
                          preproc_fn=preproc_fn,
                          image_shape=input_shape,
                          batch_size=16,
                          dataset_len=args.dataset_len,
                          shuffle=False,
                          repeat=False)

    input_x_list = []
    input_y_list = []
    for input_batch_x, input_batch_y, _ in input_data:
        input_x_list.append(input_batch_x["input_tensor"].numpy())
        input_y_list.append(input_batch_y["target"].numpy())
    input_x_arr = np.concatenate(input_x_list,axis=0)
    input_y_arr = np.concatenate(input_y_list,axis=0)
    
    # Load model
    fine_tuned_model = model_zoo.get_model(
        "coffee_net_v1", input_shape=input_shape, transfer_learning=False)    

    

    # Build and load trained model
    fine_tuned_model.load_weights(filepath=args.ckpt_path)
    logger.info(fine_tuned_model.summary())

    # From a trained model, build another to act as a feature extractor
    feature_extractor = tf.keras.Model(inputs=fine_tuned_model.input,
                                       outputs=fine_tuned_model.get_layer(args.layer_name).output)

    projector_dir = os.path.join(args.output_dir, "tensorboard", "projector")
    os.makedirs(projector_dir, exist_ok=True)
    metadata_file = open(os.path.join(
        projector_dir, "metadata_classes.tsv"), "w")
    # metadata_file.write("Class\n") # If you have only one feature, you don't have to specify a header

    # Extract embeddings and Save label metadata
    images_list = []
    feature_vectors = []

    for image_label, image_np in tqdm(zip(input_y_arr, input_x_arr)):

        images_list.append(image_np)
        image_tensor_preproc = np.expand_dims(image_np, axis=0)
        image_embedding = np.squeeze(feature_extractor(
            image_tensor_preproc.astype(np.float32)))

        feature_vectors.append(image_embedding.tolist())
        metadata_file.write('{}\n'.format(image_label))
    metadata_file.close()

    feature_vectors = np.array(feature_vectors)
    images_arr = np.array(images_list)

    # Create sprite to be displayed at tensorboard
    sprite = images_to_sprite(images_arr)
    cv2.imwrite(os.path.join(projector_dir, 'sprite.png'), sprite)

    with tf.compat.v1.Session() as sess:

        # Save embeddings as a .ckpt file
        features = tf.Variable(feature_vectors, name='features')
        sess.run(features.initializer)
        saver = tf.compat.v1.train.Saver([features])
        saver.save(sess, os.path.join(projector_dir, 'features.ckpt'))

        # Create projector
        config = projector.ProjectorConfig()
        embedding = config.embeddings.add()
        embedding.tensor_name = features.name
        embedding.metadata_path = os.path.join(
            projector_dir, 'metadata_classes.tsv')

        embedding.sprite.image_path = os.path.join(projector_dir, 'sprite.png')
        embedding.sprite.single_image_dim.extend(
            [images_arr.shape[1], images_arr.shape[2], images_arr.shape[3]])
        projector.visualize_embeddings(
            tf.compat.v1.summary.FileWriter(projector_dir), config)
