
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

    reg_val = 1e-4

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

        _max_pool = tf.keras.layers.MaxPool2D(
            pool_size=(2, 2), name="max_pool_{}".format(block_name))(_batch_norm)

        return _max_pool

    input_tensor = tf.keras.layers.Input(shape=input_shape, name="input_1")

    block_1 = _coffee_block(input_tensor, n_filters=32, block_name="1")
    block_2 = _coffee_block(block_1, n_filters=64, block_name="2")
    block_3 = _coffee_block(block_2, n_filters=128, block_name="3")
    block_4 = _coffee_block(block_3, n_filters=256, block_name="4")
    block_5 = _coffee_block(block_4, n_filters=512, block_name="5")

    global_avg_pool = tf.keras.layers.GlobalAveragePooling2D(name="avg_pool")(block_5)
    flatten = tf.keras.layers.Flatten(name="flatten")(global_avg_pool)
    fc1 = tf.keras.layers.Dense(64,
                                activation='relu',
                                kernel_regularizer=tf.keras.regularizers.L1L2(
                                    l1=reg_val, l2=reg_val),
                                bias_regularizer=tf.keras.regularizers.L1L2(
                                    l1=reg_val, l2=reg_val),
                                name="head_dense_1")(flatten)
    dropout = tf.keras.layers.Dropout(rate=0.3,name="Dropout")(fc1)
    fc2 = tf.keras.layers.Dense(2, activation='softmax', name='target')(dropout)

    return tf.keras.Model(inputs=input_tensor, outputs=fc2)
