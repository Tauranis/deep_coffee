
import tensorflow as tf


def coffee_net_v1(input_shape, transfer_learning=False):
    """

    Input 

    [   Conv2D_1
        Conv2D_2
        Concat(Conv2D_1,Conv2D_2)    
        BatchNorm
        MaxPool2D
    ] X 5

    AvgPool
    Flatten    
    FC1
    Dropout
    FC2
    Softmax


    """

    reg_val = 5e-4

    def _coffee_block(input_tensor, n_filters, activation='relu', block_name="1"):
        _conv1 = tf.keras.layers.Conv2D(n_filters,
                                        kernel_size=(3, 3),
                                        strides=(1, 1),
                                        padding='same',
                                        activation=activation,
                                        kernel_regularizer=tf.keras.regularizers.L1L2(
                                            l1=reg_val, l2=reg_val),
                                        bias_regularizer=tf.keras.regularizers.L1L2(
                                            l1=reg_val, l2=reg_val),
                                        name="Conv1_{}".format(block_name)
                                        )(input_tensor)
        _conv2 = tf.keras.layers.Conv2D(n_filters,
                                        kernel_size=(3, 3),
                                        strides=(1, 1),
                                        padding='same',
                                        activation=activation,
                                        kernel_regularizer=tf.keras.regularizers.L1L2(
                                            l1=reg_val, l2=reg_val),
                                        bias_regularizer=tf.keras.regularizers.L1L2(
                                            l1=reg_val, l2=reg_val),
                                        name="Conv2_{}".format(block_name)
                                        )(_conv1)
        _residual = tf.keras.layers.Concatenate(
            name="concat_{}".format(block_name))([_conv1, _conv2])

        _batch_norm = tf.keras.layers.BatchNormalization(
            name="batch_norm_{}".format(block_name))(_residual)

        _activation = tf.keras.layers.Activation(
            activation="relu")(_batch_norm)

        _max_pool = tf.keras.layers.MaxPool2D(
            pool_size=(2, 2), name="max_pool_{}".format(block_name))(_activation)

        return _max_pool

    input_tensor = tf.keras.layers.Input(shape=input_shape, name="input_tensor")

    block_1 = _coffee_block(input_tensor, n_filters=32,
                            block_name="1", activation=None)
    block_2 = _coffee_block(block_1, n_filters=64,
                            block_name="2", activation=None)
    block_3 = _coffee_block(block_2, n_filters=128,
                            block_name="3", activation=None)
    block_4 = _coffee_block(block_3, n_filters=256,
                            block_name="4", activation=None)
    block_5 = _coffee_block(block_4, n_filters=512,
                            block_name="5", activation=None)

    global_avg_pool = tf.keras.layers.GlobalMaxPooling2D(
        name="global_max_pool")(block_5)
    dropout = tf.keras.layers.Dropout(
        rate=0.5, name="Dropout")(global_avg_pool)
    fc1 = tf.keras.layers.Dense(64,
                                activation=None,
                                kernel_regularizer=tf.keras.regularizers.l2(
                                    l=reg_val),
                                bias_regularizer=tf.keras.regularizers.l2(
                                    l=reg_val),
                                name="head_dense_1")(dropout)
    head = tf.keras.layers.BatchNormalization(name="batch_norm")(fc1)
    head = tf.keras.layers.Activation(activation="relu")(head)
    fc2 = tf.keras.layers.Dense(2, activation='softmax', name='target')(head)
    # fc2 = tf.keras.layers.Dense(1, activation='sigmoid', name='target')(head)

    return tf.keras.Model(inputs=input_tensor, outputs=fc2)
