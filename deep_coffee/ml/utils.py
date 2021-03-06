import seaborn as sns
from matplotlib import pyplot as plt
import tensorflow as tf
import numpy as np
import itertools
import io
import os
from sklearn.metrics import confusion_matrix, roc_curve, auc
import scipy.stats as stats


import matplotlib as mpl
mpl.use('Agg')


def list_tfrecords(path_regex):

    return [f.numpy().decode('utf-8') for f in tf.data.Dataset.list_files(path_regex)]


class PlotROCCurveCallback(tf.keras.callbacks.Callback):

    def __init__(self, eval_input_fn, logdir, save_freq=1):
        self.eval_input_fn = eval_input_fn
        self.summary_image_writer = tf.summary.create_file_writer(
            os.path.join(logdir, 'roc'))
        self.save_freq = save_freq

    def on_epoch_end(self, epoch, logs):

        if ((epoch % self.save_freq == 0) or (epoch == 1)):
            Y_list = []
            Y_pred_list = []

            for data_dict in self.eval_input_fn:
                X = data_dict[0]
                Y = data_dict[1]['target'].numpy().tolist()

                # Y_pred = np.squeeze(
                #     np.array(self.model.predict_on_batch(X))).tolist()
                Y_pred = np.squeeze(
                    np.array(self.model.predict_on_batch(X)))[:, 1].tolist()  # https://github.com/keras-team/keras/issues/13118

                Y_list += Y
                Y_pred_list += Y_pred
            Y_arr = np.array(Y_list)
            Y_pred_arr = np.array(Y_pred_list)

            fpr, tpr, _ = roc_curve(Y_arr, Y_pred_arr)
            roc_auc = auc(fpr, tpr)

            # Summary auc
            with self.summary_image_writer.as_default():
                tf.summary.scalar("AUC", roc_auc, step=epoch)

            figure = plt.figure(figsize=(5, 5))
            plt.plot(fpr, tpr, color='darkorange', lw=2,
                     label='ROC curve (area = %0.2f)' % roc_auc)
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve')
            plt.legend(loc="lower right")

            plt.tight_layout()

            # Write image to tensorboard
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            plt.close(figure)
            buf.seek(0)
            # Convert PNG buffer to TF image
            image = tf.image.decode_png(buf.getvalue(), channels=4)
            buf.close()

            # Add the batch dimension
            image = tf.expand_dims(image, 0)

            with self.summary_image_writer.as_default():
                tf.summary.image("ROC Curve", image, step=epoch)

            # Plot score distribution
            figure = plt.figure(figsize=(5, 5))
            Y_pos_i = Y_arr.astype(np.bool)
            Y_neg_i = np.logical_not(Y_pos_i)
            Y_pred_pos_arr = Y_pred_arr[Y_pos_i]
            Y_pred_neg_arr = Y_pred_arr[Y_neg_i]

            sns.distplot(Y_pred_pos_arr, color="skyblue",
                         label="Good Beans", norm_hist=True,kde=True,bins=15)
            sns.distplot(Y_pred_neg_arr, color="red",
                         label="Bad Beans", norm_hist=True,kde=True,bins=15)


            plt.xlim(0, 1)
            plt.legend()

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
                tf.summary.image("Score Distribution", image, step=epoch)

            plt.close('all')


class PlotConfusionMatrixCallback(tf.keras.callbacks.Callback):

    def __init__(self, eval_input_fn, class_names, thresholds, logdir):
        self.eval_input_fn = eval_input_fn
        self.class_names = class_names
        self.summary_image_writer = tf.summary.create_file_writer(
            os.path.join(logdir, 'cm'))
        self.thresholds = thresholds

    def on_epoch_end(self, epoch, logs):

        # Make predictions
        Y_list = []
        Y_pred_list = []

        for i, data_dict in enumerate(self.eval_input_fn):
            X = data_dict[0]
            Y = data_dict[1]['target'].numpy().tolist()

            Y_pred = np.array(self.model.predict_on_batch(X))[:, 1].tolist()
            # Y_pred = np.array(self.model.predict_on_batch(X)).tolist()

            Y_list += Y
            Y_pred_list += Y_pred

        Y_arr = np.array(Y_list)
        Y_pred_arr = np.array(Y_pred_list)

        for thresh in self.thresholds:

            Y_pred_arr_thresh = (Y_pred_arr > thresh).astype(np.float32)

            cm = confusion_matrix(Y_arr, Y_pred_arr_thresh)

            # Draw confusion matrix
            figure = plt.figure(figsize=(5, 5))
            plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
            plt.title("Confusion matrix")

            tick_marks = np.arange(len(self.class_names))
            plt.xticks(tick_marks, self.class_names, rotation=45)
            plt.yticks(tick_marks, self.class_names)

            # Normalize the confusion matrix.
            cm = np.around(cm.astype('float') / cm.sum(axis=1)
                           [:, np.newaxis], decimals=2)

            # Use white text if squares are dark; otherwise black.
            threshold = cm.max() / 2.0
            # threshold = 0.5
            for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
                color = "white" if cm[i, j] > threshold else "black"
                plt.text(j, i, cm[i, j],
                         horizontalalignment="center", color=color)

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
            buf.close()

            # Add the batch dimension
            image = tf.expand_dims(image, 0)

            with self.summary_image_writer.as_default():
                tf.summary.image("Confusion Matrix at threshold {}".format(
                    thresh), image, step=epoch)
            plt.close('all')
