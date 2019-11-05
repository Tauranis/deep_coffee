import argparse
import numpy as np
import datetime
import os
import sys
import tensorflow as tf
import json
import apache_beam as beam
import tensorflow_transform as tft
from tensorflow.python.lib.io import file_io

from tensorflow_transform.beam import impl
from tensorflow_transform.tf_metadata import dataset_schema
from tensorflow_transform.tf_metadata import dataset_metadata
from tensorflow_transform.tf_metadata import schema_utils
from tensorflow_transform.coders import example_proto_coder
from tensorflow_transform.tf_metadata import metadata_io
from tensorflow_transform.beam.tft_beam_io import transform_fn_io, beam_metadata_io

try:
    try:
        from apache_beam.options.pipeline_options import PipelineOptions
    except ImportError:
        from apache_beam.utils.pipeline_options import PipelineOptions
except ImportError:
    from apache_beam.utils.options import PipelineOptions

import logging
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

CLASS_ID_BAD_BEAN = 0
CLASS_ID_GOOD_BEAN = 1

# PREPROC_FN = {
#     "densenet": tf.keras.applications.densenet.preprocess_input,
#     "inception_resnet_v2": tf.keras.applications.inception_resnet_v2.preprocess_input,
#     "inception_v3": tf.keras.applications.inception_v3.preprocess_input,
#     "mobilenet": tf.keras.applications.mobilenet.preprocess_input,
#     "mobilenet_v2": tf.keras.applications.mobilenet_v2.preprocess_input,
#     "nasnet": tf.keras.applications.nasnet.preprocess_input,
#     "resnet": tf.keras.applications.resnet.preprocess_input,
#     "resnet50": tf.keras.applications.resnet50.preprocess_input,
#     "resnet_v2": tf.keras.applications.resnet_v2.preprocess_input,
#     "vgg16": tf.keras.applications.vgg16.preprocess_input,
#     "vgg19": tf.keras.applications.vgg19.preprocess_input,
#     "xception": tf.keras.applications.xception.preprocess_input
# }


# def _preprocess_fn(features, preprocessing_fn, new_shape):
def _preprocess_fn(features, new_shape):

    # def __preprocess_image(_image_bytes):
    #     __image_tensor = tf.io.decode_jpeg(_image_bytes, channels=3)
    #     __image_tensor = tf.image.resize(__image_tensor, size=new_shape)
    #     __image_tensor = preprocessing_fn(__image_tensor)
    #     # __image_tensor = tf.image.convert_image_dtype(tf.image.resize(__image_tensor, size=new_shape),dtype=tf.uint8)
    #     # __image_tensor = tf.io.encode_jpeg(__image_tensor, quality=100)
    #     return __image_tensor
    # image_tensor = tf.map_fn(
    #     __preprocess_image, features['image_bytes'], dtype=tf.float32)

    def __preprocess_image(_image_bytes):
        __image_tensor = tf.io.decode_jpeg(_image_bytes, channels=3)        
        __image_tensor = tf.image.convert_image_dtype(tf.image.resize(__image_tensor, size=new_shape),dtype=tf.uint8)
        __image_tensor = tf.io.encode_jpeg(__image_tensor, quality=95)
        return __image_tensor

    image_tensor = tf.map_fn(
        __preprocess_image, features['image_bytes'], dtype=tf.string)

    output_features = {
        "image_preprocessed": image_tensor,
        "target": features["target"]
    }

    return output_features


def _get_feature_spec():

    schema_dict = {
        'image_bytes': tf.io.FixedLenFeature(shape=[], dtype=tf.string, default_value=None),
        'target': tf.io.FixedLenFeature(shape=[], dtype=tf.float32, default_value=None)
    }

    schema = dataset_metadata.DatasetMetadata(
        schema_utils.schema_from_feature_spec(schema_dict))
    return schema


class ReadImageDoFn(beam.DoFn):

    def __init__(self):
        super(ReadImageDoFn, self).__init__()

    def process(self, element):
        filename = element[0]
        target = element[1]

        image_bytes = tf.io.gfile.GFile(filename, mode='rb').read()

        yield {
            'image_bytes': image_bytes,
            'target': target
        }


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--tft_artifacts_dir', required=True)

    parser.add_argument('--good_beans_dir', required=True)
    parser.add_argument('--good_beans_list_train', required=True)
    parser.add_argument('--good_beans_list_eval', required=True)
    parser.add_argument('--good_beans_list_test', required=True)

    parser.add_argument('--bad_beans_dir', required=True)
    parser.add_argument('--bad_beans_list_train', required=True)
    parser.add_argument('--bad_beans_list_eval', required=True)
    parser.add_argument('--bad_beans_list_test', required=True)

    parser.add_argument('--image_dim', required=False, default=224, type=int)
    parser.add_argument('--ext', type=str, default='jpg')
    # parser.add_argument('--network', dest='network',
    #                     type=str, required=False, default=None)
    parser.add_argument('--temp-dir', dest='temp_dir',
                        required=False, default='/tmp')
    known_args, pipeline_args = parser.parse_known_args()

    good_beans_list_train = [(os.path.join(known_args.good_beans_dir, f), CLASS_ID_GOOD_BEAN) for f in tf.io.gfile.GFile(
        known_args.good_beans_list_train).read().split("\n")]
    good_beans_list_eval = [(os.path.join(known_args.good_beans_dir, f), CLASS_ID_GOOD_BEAN) for f in tf.io.gfile.GFile(
        known_args.good_beans_list_eval).read().split("\n")]
    good_beans_list_test = [(os.path.join(known_args.good_beans_dir, f), CLASS_ID_GOOD_BEAN) for f in tf.io.gfile.GFile(
        known_args.good_beans_list_test).read().split("\n")]

    bad_beans_list_train = [(os.path.join(known_args.bad_beans_dir, f), CLASS_ID_BAD_BEAN) for f in tf.io.gfile.GFile(
        known_args.bad_beans_list_train).read().split("\n")]
    bad_beans_list_eval = [(os.path.join(known_args.bad_beans_dir, f), CLASS_ID_BAD_BEAN) for f in tf.io.gfile.GFile(
        known_args.bad_beans_list_eval).read().split("\n")]
    bad_beans_list_test = [(os.path.join(known_args.bad_beans_dir, f), CLASS_ID_BAD_BEAN) for f in tf.io.gfile.GFile(
        known_args.bad_beans_list_test).read().split("\n")]

    list_train = good_beans_list_train + bad_beans_list_train
    list_eval = good_beans_list_eval + bad_beans_list_eval
    list_test = good_beans_list_test + bad_beans_list_test

    train_tfrecord_path = os.path.join(known_args.output_dir, 'train')

    # try:
    #     preproc_fn = PREPROC_FN[known_args.network]
    # except KeyError:
    #     logger.error(
    #         "Unknown preprocessor for network {}".format(known_args.network))
    #     import sys
    #     sys.exit(0)

    pipeline_options = PipelineOptions(flags=pipeline_args)

    # Preprocess dataset
    with beam.Pipeline(options=pipeline_options) as pipeline:
        with impl.Context(known_args.temp_dir):

            schema = _get_feature_spec()

            # Process training data
            pipe_train = (pipeline | beam.Create(list_train))

            # _ = pipe_train | "PRINT pipe_train" >> beam.Map(print)

            train_data = (pipe_train |
                          "Read Images - Train" >> beam.ParDo(ReadImageDoFn()))

            transformed_train, transform_fn = ((train_data, schema) |
                                               "Analyze and Transform - Train" >> impl.AnalyzeAndTransformDataset(
                                                   lambda t: _preprocess_fn(t,new_shape=(known_args.image_dim, known_args.image_dim))))
            transformed_train_data, transformed_train_metadata = transformed_train

            _ = transformed_train_data | 'Write TFrecords - train' >> beam.io.tfrecordio.WriteToTFRecord(
                file_path_prefix=train_tfrecord_path,
                file_name_suffix=".tfrecords",
                num_shards=20,
                coder=example_proto_coder.ExampleProtoCoder(transformed_train_metadata.schema))

            # # Process evaluation data

            # orders_by_date_eval = (raw_data_eval |
            #                        "Merge SKUs - eval" >> beam.CombineGlobally(GroupItemsByDate(community_area_list, (split_datetime, end_datetime))))

            # ts_windows_eval = (orders_by_date_eval | "Extract timeseries windows - eval" >>
            #                    beam.ParDo(ExtractRawTimeseriesWindow(known_args.window_size)) |
            #                    "Fusion breaker eval" >> beam.Reshuffle())

            # norm_ts_windows_eval = (((ts_windows_eval, schema), transform_fn) |
            #                         "Transform - eval" >> impl.TransformDataset())

            # norm_ts_windows_eval_data, norm_ts_windows_eval_metadata = norm_ts_windows_eval

            # _ = norm_ts_windows_eval_data | 'Write TFrecords - eval' >> beam.io.tfrecordio.WriteToTFRecord(
            #     file_path_prefix=eval_tfrecord_path,
            #     file_name_suffix=".tfrecords",
            #     coder=example_proto_coder.ExampleProtoCoder(norm_ts_windows_eval_metadata.schema))

            # # Dump raw eval set for further tensorflow model analysis
            # _ = ts_windows_eval | 'Write TFrecords - eval raw' >> beam.io.tfrecordio.WriteToTFRecord(
            #     file_path_prefix=eval_raw_tfrecord_path,
            #     file_name_suffix=".tfrecords",
            #     coder=example_proto_coder.ExampleProtoCoder(schema.schema))

            # # Dump transformation graph
            # _ = transform_fn | 'Dump Transform Function Graph' >> transform_fn_io.WriteTransformFn(
            #     known_args.tft_artifacts_dir)
