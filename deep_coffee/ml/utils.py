import seaborn as sns
from matplotlib import pyplot as plt
import tensorflow as tf
import numpy as np
import itertools
import io
import os
from sklearn.metrics import confusion_matrix

import matplotlib as mpl
mpl.use('Agg')


def list_tfrecords(path_regex):
    return [f.numpy().decode('utf-8') for f in tf.data.Dataset.list_files(path_regex)]


class PlotConfusionMatrixCallback(tf.keras.callbacks.Callback):

    def __init__(self, eval_input_fn, class_names, logdir):
        self.eval_input_fn = eval_input_fn
        self.class_names = class_names
        self.summary_image_writer = tf.summary.create_file_writer(
            os.path.join(logdir, 'cm'))

    def on_epoch_end(self, epoch, logs):

        # Make predictions
        Y_list = []
        Y_pred_list = []

        for i, data_dict in enumerate(self.eval_input_fn):
            X = data_dict[0]
            Y = data_dict[1]['target'].numpy().tolist()

            Y_pred = np.argmax(self.model.predict(X), axis=1).tolist()

            Y_list += Y
            Y_pred_list += Y_pred

        Y_arr = np.array(Y_list)
        Y_pred_arr = np.array(Y_pred_list)
        cm = confusion_matrix(Y_arr, Y_pred_arr)

        # Draw confusion matrix
        figure = plt.figure(figsize=(5, 5))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues,vmin=0.1, vmax=0.9)
        plt.title("Confusion matrix")
        # plt.colorbar()
        tick_marks = np.arange(len(self.class_names))
        plt.xticks(tick_marks, self.class_names, rotation=45)
        plt.yticks(tick_marks, self.class_names)

        # ax = sns.heatmap(cm, annot=True, fmt="d")
        # ax.set_xlabel("Ground Truth")
        # ax.set_xlabel("Prediction")

        # Normalize the confusion matrix.
        cm = np.around(cm.astype('float') / cm.sum(axis=1)
                       [:, np.newaxis], decimals=2)

        # Use white text if squares are dark; otherwise black.
        # threshold = cm.max() / 2.0
        threshold = 0.5
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            color = "white" if cm[i, j] > threshold else "black"
            plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

        plt.ylabel('Ground Truth')
        plt.xlabel('Prediction')
        plt.tight_layout()

        # Write image to tensorboard
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close(figure)
        buf.seek(0)
        # Convert PNG buffer to TF image
        image = tf.image.decode_png(buf.getvalue(), channels=4)
        # Add the batch dimension
        image = tf.expand_dims(image, 0)

        with self.summary_image_writer.as_default():
            tf.summary.image("Confusion Matrix", image, step=epoch)
