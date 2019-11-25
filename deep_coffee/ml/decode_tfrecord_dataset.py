import tensorflow as tf
import argparse
import cv2
import os
from deep_coffee.ml.utils import list_tfrecords
from tqdm import tqdm

from tensorflow_transform import TFTransformOutput
from tensorflow_transform.beam.tft_beam_io import transform_fn_io


def input_fn(tfrecords_path,
             tft_metadata,
             image_shape,
             batch_size):

    def _parse_example(proto):
            return tf.io.parse_single_example(proto, features=tft_metadata.transformed_feature_spec())    

    def _decode_sample(record):
        image_decoded = tf.image.decode_jpeg(record["image_bytes"], channels=3)
        image_decoded = tf.reshape(image_decoded, image_shape)
        record["image_decoded"] = image_decoded
        return record

    dataset = tf.data.TFRecordDataset(tfrecords_path)
    dataset = dataset.map(_parse_example)
    dataset = dataset.map(_decode_sample)
    dataset = dataset.batch(batch_size)

    return dataset


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--tfrecord_file', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--batch_size', required=False, default=32)
    parser.add_argument("--tft_artifacts_dir", required=True)
    args = parser.parse_args()

    tft_metadata_dir = os.path.join(
        args.tft_artifacts_dir, transform_fn_io.TRANSFORM_FN_DIR)
    tft_metadata = TFTransformOutput(args.tft_artifacts_dir)

    tfrecords_list = list_tfrecords(args.tfrecord_file)

    dataset = input_fn(tfrecords_list,
                       tft_metadata,
                       (224, 224, 3),
                       args.batch_size)    

    with tqdm() as pbar:
        for batch in dataset:
            # last batch maybe lesser than args.batch_size
            _batch_size = len(batch["filename"].numpy())
            for i in range(_batch_size):
                filename = os.path.join(args.output_dir, batch["target_name"].numpy()[
                                        i].decode("utf-8"), "{}_{}".format(str(int(batch["target"].numpy()[i])), batch["filename"].numpy()[i].decode("utf-8")))
                bgr_image = cv2.cvtColor(batch["image_decoded"].numpy()[
                                         i], cv2.COLOR_RGB2BGR)
                cv2.imwrite(filename, bgr_image)
                pbar.update()
