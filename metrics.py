'''
Author: Zhiyuan Yan
Email: yanzhiyuan1114@gmail.com
Time: 2023-04-14
'''

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d

def compute_eer(y_true, y_score):
    """
    Compute the equal error rate (EER) given true labels and predicted scores.

    Args:
        y_true (array): True binary labels.
        y_score (array): Predicted scores.

    Returns:
        float: The EER.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=1)
    fnr = 1 - tpr
    eer_threshold = thresholds[np.nanargmin(np.abs(fnr - fpr))]
    eer = brentq(lambda x: 1. - interp1d(fpr, tpr)(x) - x, 0., 1.)
    return eer
