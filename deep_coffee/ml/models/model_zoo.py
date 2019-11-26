from deep_coffee.ml.models.coffee_net import coffee_net_v1
import sys

import tensorflow as tf
from tensorflow.keras import backend as K

import logging
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def model_head(input_shape, backend_model, transfer_learning=True):

    reg_val = 5e-4

    K.set_learning_phase(0)  # https://github.com/keras-team/keras/pull/9965
    base_model = backend_model(
        include_top=True, input_tensor=tf.keras.layers.Input(shape=input_shape, name="input_tensor"), layers=tf.keras.layers)
    base_model.trainable = not transfer_learning
    K.set_learning_phase(1)
    head = tf.keras.layers.Dropout(0.5, name="dropout_head")(base_model.output)
    head = tf.keras.layers.Dense(
        64, activation=None, kernel_regularizer=tf.keras.regularizers.l2(l=reg_val))(head)
    head = tf.keras.layers.BatchNormalization()(head)
    head = tf.keras.layers.Activation(activation="relu")(head)

    head = tf.keras.layers.Dense(2,
                                 activation='softmax',
                                 name='target')(head)

    return tf.keras.Model(inputs=base_model.input, outputs=head)


def mobilenet(input_shape, transfer_learning=True):
    return model_head(input_shape, tf.keras.applications.MobileNet, transfer_learning)


def vgg16(input_shape, transfer_learning=True):
    return model_head(input_shape, tf.keras.applications.VGG16, transfer_learning)


def resnet50(input_shape, transfer_learning=True):
    return model_head(input_shape, tf.keras.applications.ResNet50, transfer_learning)


MODEL_ZOO = {
    "mobilenet": mobilenet,
    "resnet50": resnet50,
    "vgg16": vgg16,
    "coffee_net_v1": coffee_net_v1,
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
