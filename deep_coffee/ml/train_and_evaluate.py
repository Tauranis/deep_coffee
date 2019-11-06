
import logging
from deep_coffee.ml.models import model_zoo
from deep_coffee.ml.utils import list_tfrecords
from tensorflow.keras import backend as K
from tensorflow.python.lib.io import file_io
from tensorflow_transform import TFTransformOutput
from tensorflow_transform.tf_metadata import schema_utils
from tensorflow_transform.tf_metadata import dataset_metadata
from tensorflow_transform.tf_metadata import dataset_schema
from tensorflow_transform.tf_metadata import metadata_io
from tensorflow_transform.beam.tft_beam_io import transform_fn_io
import tensorflow as tf
import argparse
import yaml
import os

import multiprocessing
N_CORES = multiprocessing.cpu_count()


logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def input_fn(tfrecords_path,
             tft_metadata,
             image_shape,
             batch_size=8):
    """ Train input function
        Create and parse dataset from tfrecords shards with TFT schema
    """

    def _parse_example(proto):
        return tf.io.parse_single_example(proto, features=tft_metadata.transformed_feature_spec())

    def _split_XY(example):
        X = {}
        Y = {}

        image_tensor = tf.io.decode_jpeg(example['image_preprocessed'], channels=3)
        image_tensor = tf.reshape(image_tensor,image_shape)
        X['input_1'] = image_tensor / 255
        Y['target'] = example['target']

        return X, Y

    num_parallel_calls = N_CORES-1
    if num_parallel_calls <= 0:
        num_parallel_calls = 1

    dataset = tf.data.TFRecordDataset(tfrecords_path, compression_type="")
    dataset = dataset.map(_parse_example,
                          num_parallel_calls=num_parallel_calls)
    dataset = dataset.map(_split_XY, num_parallel_calls=num_parallel_calls)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.shuffle(buffer_size=batch_size * 10)
    dataset = dataset.repeat()

    return dataset


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--tft_artifacts_dir', required=True)
    parser.add_argument('--input_dim', required=False, default=224, type=int)
    parser.add_argument('--config_file', required=True)
    parser.add_argument('--transfer_learning',
                        required=False, action='store_true')
    args = parser.parse_args()

    input_shape = [args.input_dim, args.input_dim, 3]

    config = yaml.load(tf.io.gfile.GFile(args.config_file).read())

    logger.info('Load tfrecords...')
    tfrecords_train = list_tfrecords(config['tfrecord_train'])
    logger.info(tfrecords_train[:3])
    tfrecords_eval = list_tfrecords(config['tfrecord_eval'])
    logger.info(tfrecords_eval[:3])
    tfrecords_test = list_tfrecords(config['tfrecord_test'])
    logger.info(tfrecords_test[:3])

    tft_metadata_dir = os.path.join(
        args.tft_artifacts_dir, transform_fn_io.TRANSFORM_FN_DIR)
    tft_metadata = TFTransformOutput(args.tft_artifacts_dir)

    model = model_zoo.get_model(
        config['network_name'], input_shape=input_shape, transfer_learning=config['transfer_learning'])
    logger.info(model.summary())

    model.compile(optimizer=tf.keras.optimizers.Adam(lr=config['learning_rate']),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    train_spec = tf.estimator.TrainSpec(
        input_fn=lambda: input_fn(tfrecords_train,
                                  tft_metadata,
                                  input_shape,
                                  config['batch_size']))
    # max_steps=steps_per_epoch_train*args.epochs)

    eval_spec = tf.estimator.EvalSpec(
        input_fn=lambda: input_fn(tfrecords_eval,
                                  tft_metadata,
                                  input_shape,
                                  config['batch_size']))
    # steps=steps_per_epoch_eval*args.epochs)

    run_config = tf.estimator.RunConfig(
        model_dir=args.output_dir,
        save_summary_steps=1000,
        save_checkpoints_steps=1000,
        keep_checkpoint_max=1
    )

    model_estimator = tf.keras.estimator.model_to_estimator(
        keras_model=model, config=run_config)

    logger.info('Train')
    tf.estimator.train_and_evaluate(estimator=model_estimator,
                                    train_spec=train_spec,
                                    eval_spec=eval_spec)
