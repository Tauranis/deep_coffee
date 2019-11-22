import multiprocessing
from tensorflow_transform.beam.tft_beam_io import transform_fn_io, beam_metadata_io
from tensorflow_transform.tf_metadata import metadata_io
from tensorflow_transform.coders import example_proto_coder
from tensorflow_transform.tf_metadata import schema_utils
from tensorflow_transform.tf_metadata import dataset_metadata
from tensorflow_transform.tf_metadata import dataset_schema
from tensorflow_transform.beam import impl
import tensorflow_transform as tft
import apache_beam as beam
import tensorflow as tf
import logging
import argparse
import os
# Disable GPU, otherwise TFT will raise an error during the process
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


N_CORES = multiprocessing.cpu_count()

try:
    try:
        from apache_beam.options.pipeline_options import PipelineOptions, DirectOptions
    except ImportError:
        from apache_beam.utils.pipeline_options import PipelineOptions, DirectOptions
except ImportError:
    from apache_beam.utils.options import PipelineOptions, DirectOptions

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

CLASS_ID_BAD_BEAN = 0
CLASS_ID_GOOD_BEAN = 1


def _preprocess_fn(features, new_shape):

    def __preprocess_image(_image_bytes):
        __image_tensor = tf.io.decode_jpeg(_image_bytes, channels=3)
        __image_tensor = tf.image.convert_image_dtype(
            tf.image.resize(__image_tensor, size=new_shape)/255.0, dtype=tf.uint8)
        __image_tensor = tf.io.encode_jpeg(__image_tensor, quality=95)
        return __image_tensor

    _image_tensor = tf.map_fn(
        __preprocess_image, features['image_bytes'], dtype=tf.string)

    _output_features = {
        "image_preprocessed": _image_tensor,
        "target": features["target"]
    }

    return _output_features


def _get_feature_spec():

    _schema_dict = {
        'image_bytes': tf.io.FixedLenFeature(shape=[], dtype=tf.string, default_value=None),
        'target': tf.io.FixedLenFeature(shape=[], dtype=tf.float32, default_value=None)
    }

    schema = dataset_metadata.DatasetMetadata(
        schema_utils.schema_from_feature_spec(_schema_dict))
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

    parser.add_argument('--n_shards', required=False, default=10, type=int)

    parser.add_argument('--temp-dir', dest='temp_dir',
                        required=False, default='/tmp')
    known_args, pipeline_args = parser.parse_known_args()

    good_beans_list_train = [(os.path.join(known_args.good_beans_dir, f), CLASS_ID_GOOD_BEAN) for f in tf.io.gfile.GFile(
        known_args.good_beans_list_train).read().split("\n")[:-1]]
    good_beans_list_eval = [(os.path.join(known_args.good_beans_dir, f), CLASS_ID_GOOD_BEAN) for f in tf.io.gfile.GFile(
        known_args.good_beans_list_eval).read().split("\n")[:-1]]
    good_beans_list_test = [(os.path.join(known_args.good_beans_dir, f), CLASS_ID_GOOD_BEAN) for f in tf.io.gfile.GFile(
        known_args.good_beans_list_test).read().split("\n")[:-1] ]

    bad_beans_list_train = [(os.path.join(known_args.bad_beans_dir, f), CLASS_ID_BAD_BEAN) for f in tf.io.gfile.GFile(
        known_args.bad_beans_list_train).read().split("\n")[:-1]]
    bad_beans_list_eval = [(os.path.join(known_args.bad_beans_dir, f), CLASS_ID_BAD_BEAN) for f in tf.io.gfile.GFile(
        known_args.bad_beans_list_eval).read().split("\n")[:-1]]
    bad_beans_list_test = [(os.path.join(known_args.bad_beans_dir, f), CLASS_ID_BAD_BEAN) for f in tf.io.gfile.GFile(
        known_args.bad_beans_list_test).read().split("\n")[:-1]]

    list_train = good_beans_list_train + bad_beans_list_train
    list_eval = good_beans_list_eval + bad_beans_list_eval
    list_test = good_beans_list_test + bad_beans_list_test

    train_tfrecord_path = os.path.join(known_args.output_dir, 'train')
    eval_tfrecord_path = os.path.join(known_args.output_dir, 'eval')
    test_tfrecord_path = os.path.join(known_args.output_dir, 'test')

    #logger.info(list_train)
    #logger.info(list_eval)
    #logger.info(list_test)

    #print(list_train[:10])
    #import sys
    #sys.exit(0)

    pipeline_options = PipelineOptions(flags=pipeline_args)
    pipeline_options.view_as(DirectOptions).direct_num_workers = 1 if (
        N_CORES-2) <= 0 else (N_CORES-2)

    # Preprocess dataset
    with beam.Pipeline(options=pipeline_options) as pipeline:
        with impl.Context(known_args.temp_dir):

            schema = _get_feature_spec()

            # Process training data
            train_data = (pipeline |
                          "Create train list" >> beam.Create(list_train) |
                          "Read Images - Train" >> beam.ParDo(ReadImageDoFn()))

            transformed_train, transform_fn = ((train_data, schema) |
                                               "Analyze and Transform - Train" >> impl.AnalyzeAndTransformDataset(
                                                   lambda t: _preprocess_fn(t, new_shape=(known_args.image_dim, known_args.image_dim))))
            transformed_train_data, transformed_train_metadata = transformed_train

            _ = transformed_train_data | 'Write TFrecords - train' >> beam.io.tfrecordio.WriteToTFRecord(
                file_path_prefix=train_tfrecord_path,
                file_name_suffix=".tfrecords",
                num_shards=known_args.n_shards,
                coder=example_proto_coder.ExampleProtoCoder(transformed_train_metadata.schema))

            # Process evaluation data
            eval_data = (pipeline |
                         "Create eval list" >> beam.Create(list_eval) |
                         "Read Images - Eval" >> beam.ParDo(ReadImageDoFn()))

            transformed_eval = (((eval_data, schema), transform_fn) |
                                "Transform - Eval" >> impl.TransformDataset())

            transformed_eval_data, transformed_eval_metadata = transformed_eval

            _ = transformed_eval_data | 'Write TFrecords - eval' >> beam.io.tfrecordio.WriteToTFRecord(
                file_path_prefix=eval_tfrecord_path,
                file_name_suffix=".tfrecords",
                num_shards=known_args.n_shards,
                coder=example_proto_coder.ExampleProtoCoder(transformed_eval_metadata.schema))

            # Process test data
            test_data = (pipeline |
                         "Create test list" >> beam.Create(list_test) |
                         "Read Images - Test" >> beam.ParDo(ReadImageDoFn()))

            transformed_test = (((test_data, schema), transform_fn) |
                                "Transform - Test" >> impl.TransformDataset())

            transformed_test_data, transformed_test_metadata = transformed_test

            _ = transformed_test_data | 'Write TFrecords - test' >> beam.io.tfrecordio.WriteToTFRecord(
                file_path_prefix=test_tfrecord_path,
                file_name_suffix=".tfrecords",
                num_shards=known_args.n_shards,
                coder=example_proto_coder.ExampleProtoCoder(transformed_test_metadata.schema))

            # Dump transformation graph
            _ = transform_fn | 'Dump Transform Function Graph' >> transform_fn_io.WriteTransformFn(
                known_args.tft_artifacts_dir)
