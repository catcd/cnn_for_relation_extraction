from sklearn.metrics import precision_recall_fscore_support
import numpy as np


def calculate_f_measure(y_true, y_pred):
    _y_true = np.array([(0 if i[0] == 1 else 1) for i in y_true])
    print(_y_true)
    return precision_recall_fscore_support(_y_true, y_pred, average='binary')
