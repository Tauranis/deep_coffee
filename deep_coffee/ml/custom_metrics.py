
import tensorflow as tf

from tensorflow.python.keras.utils import metrics_utils
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.keras import backend as K
from tensorflow.python.keras.utils.generic_utils import to_list

import numpy as np

class CustomRecall(tf.keras.metrics.Metric):
  
  def __init__(self,
               threshold=0.5,               
               class_id=None,
               name=None,
               dtype=None):
    
    super(CustomRecall, self).__init__(name=name, dtype=dtype)
    self.threshold = [threshold]
    self.class_id = class_id
        
    self.true_positives = self.add_weight(
        'true_positives',
        shape=(len(self.thresholds),),
        initializer=init_ops.zeros_initializer)
    self.false_negatives = self.add_weight(
        'false_negatives',
        shape=(len(self.thresholds),),
        initializer=init_ops.zeros_initializer)

  def update_state(self, y_true, y_pred, sample_weight=None):    
    
    import sys
    print(y_pred)
    print(y_pred.shape)
    print("\n\n\n\n\n DDDDDDDDDDD \n\n\n\n")
    sys.exit(0)

    return metrics_utils.update_confusion_matrix_variables(
        {
            metrics_utils.ConfusionMatrix.TRUE_POSITIVES: self.true_positives,
            metrics_utils.ConfusionMatrix.FALSE_NEGATIVES: self.false_negatives
        },
        y_true,
        y_pred,
        threshold=self.threshold,        
        )

  def result(self):
    result = math_ops.div_no_nan(self.true_positives,
                                 self.true_positives + self.false_negatives)
    return result[0] if len(self.thresholds) == 1 else result

  def reset_states(self):
    num_thresholds = len(to_list(self.thresholds))
    K.batch_set_value(
        [(v, np.zeros((num_thresholds,))) for v in self.variables])