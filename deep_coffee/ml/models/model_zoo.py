import sys

import tensorflow as tf

import logging
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def mobilenet(input_shape, transfer_learning=True):
    base_model = tf.keras.applications.MobileNet(
        include_top=False, input_shape=input_shape)
    base_model.trainable = not transfer_learning

    head = tf.keras.layers.MaxPooling2D(
        pool_size=7, name="head_maxpool2D")(base_model.output)
    head = tf.keras.layers.Flatten(name="head_flatten")(head)
    head = tf.keras.layers.Dense(64,
                                 activation='relu',
                                 kernel_regularizer=tf.keras.regularizers.L1L2(
                                     l1=1e-5, l2=1e-5),
                                 bias_regularizer=tf.keras.regularizers.L1L2(
                                     l1=1e-5, l2=1e-5),
                                 name="head_dense_1")(head)
    head = tf.keras.layers.Dense(2, activation='softmax', name='target')(head)

    return tf.keras.Model(inputs=base_model.input, outputs=head)

def vgg16(input_shape, transfer_learning=True):
    base_model = tf.keras.applications.VGG16(
        include_top=False, input_shape=input_shape)
    base_model.trainable = not transfer_learning

    head = tf.keras.layers.MaxPooling2D(
        pool_size=7, name="head_maxpool2D")(base_model.output)
    head = tf.keras.layers.Flatten(name="head_flatten")(head)
    head = tf.keras.layers.Dense(64,
                                 activation='relu',
                                 kernel_regularizer=tf.keras.regularizers.L1L2(
                                     l1=1e-5, l2=1e-5),
                                 bias_regularizer=tf.keras.regularizers.L1L2(
                                     l1=1e-5, l2=1e-5),
                                 name="head_dense_1")(head)
    head = tf.keras.layers.Dense(2, activation='softmax', name='target')(head)

    return tf.keras.Model(inputs=base_model.input, outputs=head)

def resnet50(input_shape, transfer_learning=True):
    base_model = tf.keras.applications.ResNet50(
        include_top=False, input_shape=input_shape)
    base_model.trainable = not transfer_learning

    head = tf.keras.layers.MaxPooling2D(
        pool_size=7, name="head_maxpool2D")(base_model.output)
    head = tf.keras.layers.Flatten(name="head_flatten")(head)
    head = tf.keras.layers.Dense(64,
                                 activation='relu',
                                 kernel_regularizer=tf.keras.regularizers.L1L2(
                                     l1=1e-5, l2=1e-5),
                                 bias_regularizer=tf.keras.regularizers.L1L2(
                                     l1=1e-5, l2=1e-5),
                                 name="head_dense_1")(head)
    head = tf.keras.layers.Dense(2, activation='softmax', name='target')(head)

    return tf.keras.Model(inputs=base_model.input, outputs=head)

MODEL_ZOO = {
    "mobilenet": mobilenet,
    "resnet50": resnet50,
    "vgg16": vgg16,
    # "vgg19": tf.keras.applications.VGG19
}


def get_model(model_name, **kw):
    """ Get a Tensorflow model """
    if model_name not in MODEL_ZOO:
        logger.error('Model {} does not exist')
        sys.exit(1)

    logger.info('Loading model {}'.format(model_name))
    logger.info('kw: {}'.format(kw))
    model = MODEL_ZOO[model_name](**kw)
    return model
