
import argparse

import tensorflow as tf
import cv2
import numpy as np

import logging
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--sample_image_path", required=True)
    args = parser.parse_args()

    image_sample = np.expand_dims(cv2.resize(cv2.cvtColor(cv2.imread(
        args.sample_image_path), cv2.COLOR_BGR2RGB), (224, 224)), axis=0)/255.0
    logger.info("Image shape {}: ".format(image_sample.shape))

    model = tf.saved_model.load(args.model_path)

    prediction = model.signatures["serving_default"](
        tf.constant(image_sample, dtype=tf.float32))

    logger.info(prediction)
