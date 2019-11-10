import tensorflow as tf
import argparse
import cv2

def _parse_example(proto):
        return tf.io.parse_single_example(proto, features={
            "image_preprocessed":tf.io.FixedLenFeature([],dtype=tf.string,default_value=None),
            "target":tf.io.FixedLenFeature([],dtype=tf.float32, default_value=None)
        })

def _decode_image(record):
    image_decoded = tf.image.decode_jpeg(record["image_preprocessed"])
    image_decoded = tf.reshape(image_decoded,(224,224,3))
    record["image_preprocessed"] = image_decoded
    return record

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--tfrecord_file', required=True)
    args = parser.parse_args()

    dataset = tf.data.TFRecordDataset(args.tfrecord_file)
    dataset = dataset.map(_parse_example)
    dataset = dataset.map(_decode_image)
    dataset = dataset.batch(1)

    for batch in dataset:
        cv2.imwrite("/trained_models/sample.jpg",batch["image_preprocessed"].numpy()[0])
        break
