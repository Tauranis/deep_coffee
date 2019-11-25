
import logging
from deep_coffee.ml.models import model_zoo, preproc_zoo
from deep_coffee.ml.utils import list_tfrecords, PlotConfusionMatrixCallback, PlotROCCurveCallback
from tensorflow_transform import TFTransformOutput
from tensorflow_transform.beam.tft_beam_io import transform_fn_io
from tf_explain.callbacks.grad_cam import GradCAMCallback
import tensorflow as tf
import argparse
import yaml
import os
import datetime
import numpy as np

# tf.compat.v1.disable_eager_execution()

import multiprocessing
N_CORES = multiprocessing.cpu_count()

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

GRAD_CAM_GRID_SIZE = 16


def input_fn(tfrecords_path,
             tft_metadata,
             preproc_fn,
             image_shape,
             batch_size=8,
             shuffle=True,
             repeat=True):
    """ Train input function
        Create and parse dataset from tfrecords shards with TFT schema
    """

    def _parse_example(proto):
        return tf.io.parse_single_example(proto, features=tft_metadata.transformed_feature_spec())

    def _split_XY(example):
        X = {}
        Y = {}

        image_tensor = tf.image.decode_jpeg(
            example["image_bytes"], channels=3)
        image_tensor = tf.reshape(image_tensor, image_shape)
        image_tensor = tf.dtypes.cast(image_tensor, tf.float32)
        # X["input_1"] = preproc_fn(image_tensor)
        X["input_1"] = image_tensor/255.0
        Y["target"] = example["target"]

        return X, Y

    num_parallel_calls = None  # N_CORES-1
    # if num_parallel_calls <= 0:
    #    num_parallel_calls = 1

    dataset = tf.data.TFRecordDataset(tfrecords_path, compression_type="")
    dataset = dataset.map(_parse_example,
                          num_parallel_calls=num_parallel_calls)
    dataset = dataset.map(_split_XY, num_parallel_calls=num_parallel_calls)
    #dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    # dataset = dataset.prefetch(1) #tf.data.experimental.AUTOTUNE)

    if shuffle:
        dataset = dataset.shuffle(buffer_size=batch_size * 3)

    if repeat:
        dataset = dataset.repeat()

    return dataset


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--tft_artifacts_dir", required=True)
    parser.add_argument("--input_dim", required=False, default=224, type=int)
    parser.add_argument("--trainset_len", required=True, type=int)
    parser.add_argument("--evalset_len", required=True, type=int)
    parser.add_argument("--testset_len", required=True, type=int)
    parser.add_argument("--config_file", required=True)
    args = parser.parse_args()

    temp = tf.random.uniform([4, 32, 32, 3])  # Or tf.zeros
    tf.keras.applications.vgg16.preprocess_input(temp)

    input_shape = [args.input_dim, args.input_dim, 3]

    config = yaml.load(tf.io.gfile.GFile(args.config_file).read())

    logger.info("Load tfrecords...")
    tfrecords_train = list_tfrecords(config["tfrecord_train"])
    logger.info(tfrecords_train[:3])
    tfrecords_eval = list_tfrecords(config["tfrecord_eval"])
    logger.info(tfrecords_eval[:3])
    tfrecords_test = list_tfrecords(config["tfrecord_test"])
    logger.info(tfrecords_test[:3])

    # Load TFT metadata
    tft_metadata_dir = os.path.join(
        args.tft_artifacts_dir, transform_fn_io.TRANSFORM_FN_DIR)
    tft_metadata = TFTransformOutput(args.tft_artifacts_dir)
    preproc_fn = preproc_zoo.get_preproc_fn(config["network_name"])

    model = model_zoo.get_model(
        config["network_name"], input_shape=input_shape, transfer_learning=config["transfer_learning"])
    logger.info(model.summary())

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=config["learning_rate"]),
    # model.compile(optimizer=tf.keras.optimizers.SGD(lr=config["learning_rate"],momentum=0.9),
    # model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=config["learning_rate"]),
                    # loss="sparse_categorical_crossentropy",
                  loss="binary_crossentropy",
                  metrics=["acc",
                           tf.keras.metrics.AUC(num_thresholds=20),
                           tf.keras.metrics.Precision(thresholds=[0.1,0.25,0.5,0.75,0.9]),
                           tf.keras.metrics.Recall(thresholds=[0.1,0.25,0.5,0.75,0.9])
                           ])

    steps_per_epoch_train = args.trainset_len // config["batch_size"]
    steps_per_epoch_eval = args.evalset_len // config["batch_size"]
    steps_per_epoch_test = args.testset_len // config["batch_size"]

    datetime_now_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = os.path.join(
        args.output_dir, config["network_name"], datetime_now_str)
    ckpt_dir = os.path.join(output_dir, "model.ckpt")
    tensorboard_dir = os.path.join(output_dir, "tensorboard")

    callback_list = []

    callback_save_ckpt = tf.keras.callbacks.ModelCheckpoint(filepath=ckpt_dir,
                                                            monitor="val_loss",
                                                            save_best_only=True,
                                                            save_freq="epoch")
    # callback_list.append(callback_save_ckpt)

    callback_tensorboard = tf.keras.callbacks.TensorBoard(log_dir=tensorboard_dir,
                                                          histogram_freq=10,
                                                          write_graph=True,
                                                          write_images=False,
                                                          profile_batch=0,
                                                          update_freq="epoch")
    callback_list.append(callback_tensorboard)

    callback_early_stop = tf.keras.callbacks.EarlyStopping(monitor="val_loss",
                                                           min_delta=5e-3,
                                                           patience=10)
    # callback_list.append(callback_early_stop)

    callback_plot_cm = PlotConfusionMatrixCallback(eval_input_fn=input_fn(tfrecords_eval,
                                                                          tft_metadata,
                                                                          preproc_fn,
                                                                          input_shape,
                                                                          config["batch_size"],
                                                                          shuffle=False,
                                                                          repeat=False),
                                                   class_names=[
                                                       "Bad Beans", "Good Beans"],
                                                   thresholds=[
                                                       0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                                                   logdir=tensorboard_dir)
    callback_list.append(callback_plot_cm)

    callback_plot_roc = PlotROCCurveCallback(eval_input_fn=input_fn(tfrecords_eval,
                                                                    tft_metadata,
                                                                    preproc_fn,
                                                                    input_shape,
                                                                    config["batch_size"],
                                                                    shuffle=False,
                                                                    repeat=False),
                                             logdir=tensorboard_dir,
                                             save_freq=1)
    callback_list.append(callback_plot_roc)

    try:

        # Gather 1 Batch for GradCAM callback
        # TODO: https://github.com/sicara/tf-explain/issues/67
        grad_cam_sample = input_fn(tfrecords_eval,
                                   tft_metadata,
                                   preproc_fn,
                                   input_shape,
                                   GRAD_CAM_GRID_SIZE*2,
                                   shuffle=False,
                                   repeat=False)
        for class_index in [0, 1]:

            grad_cam_x = []
            _buff_size = 0

            for grad_cam_sample_x, grad_cam_sample_y in grad_cam_sample:
                grad_cam_sample_x = grad_cam_sample_x["input_1"].numpy()
                grad_cam_sample_y = np.array([
                    int(v) for v in grad_cam_sample_y["target"].numpy()])
                class_index_i = np.where(grad_cam_sample_y == class_index)[0]
                grad_cam_x.append(grad_cam_sample_x[class_index_i])

                _buff_size += len(class_index_i)
                if _buff_size >= GRAD_CAM_GRID_SIZE:
                    break
            grad_cam_x = np.concatenate(grad_cam_x, axis=0)[
                :GRAD_CAM_GRID_SIZE]

            for grad_cam_layer in config["grad_cam_layers"]:
                callback_grad_cam = GradCAMCallback(
                    validation_data=(grad_cam_x, None),
                    layer_name=grad_cam_layer,
                    class_index=class_index,
                    output_dir=os.path.join(
                        tensorboard_dir, "grad_cam", grad_cam_layer, "class_{}".format(class_index)),
                )
                callback_list.append(callback_grad_cam)
    except KeyError:
        pass

    model.fit(x=input_fn(tfrecords_path=tfrecords_train,
                         tft_metadata=tft_metadata,
                         preproc_fn=preproc_fn,
                         image_shape=input_shape,
                         batch_size=config["batch_size"]),
              validation_data=input_fn(tfrecords_eval,
                                       tft_metadata,
                                       preproc_fn,
                                       input_shape,
                                       config["batch_size"],
                                       shuffle=False),
              steps_per_epoch=steps_per_epoch_train,
              validation_steps=steps_per_epoch_eval,
              epochs=config["epochs"],
              callbacks=callback_list,
            #   callbacks=None,
              class_weight={  # TODO: parameterize
                  0: 2.12,
                  1: 0.65
    })

    # train_spec = tf.estimator.TrainSpec(
    #     input_fn=lambda: input_fn(tfrecords_train,
    #                               tft_metadata,
    #                               input_shape,
    #                               config["batch_size"]))
    # # max_steps=steps_per_epoch_train*args.epochs)

    # eval_spec = tf.estimator.EvalSpec(
    #     input_fn=lambda: input_fn(tfrecords_eval,
    #                               tft_metadata,
    #                               input_shape,
    #                               config["batch_size"]))
    # # steps=steps_per_epoch_eval*args.epochs)

    # run_config = tf.estimator.RunConfig(
    #     model_dir=args.output_dir,
    #     save_summary_steps=1000,
    #     save_checkpoints_steps=1000,
    #     keep_checkpoint_max=1
    # )

    # model_estimator = tf.keras.estimator.model_to_estimator(
    #     keras_model=model, config=run_config)

    # logger.info("Train")
    # tf.estimator.train_and_evaluate(estimator=model_estimator,
    #                                 train_spec=train_spec,
    #                                 eval_spec=eval_spec)
