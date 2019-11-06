import tensorflow as tf


def list_tfrecords(path_regex):
    return [f.numpy().decode('utf-8') for f in tf.data.Dataset.list_files(path_regex)]
