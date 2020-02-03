import numpy as np

def calc_precision_recall_f1_bacc(prediction, target):
    target_positives = np.sum(target == 1)
    target_negatives = np.sum(target == 0)
    pred_positives = np.sum(prediction)
    true_positives = np.sum((prediction == target) * (target > 0))
    true_negatives = np.sum((prediction == target) * (target == 0))

    if target_positives == 0:
        return (None, None, None, None)

    precision = (true_positives / pred_positives) if pred_positives > 0 else 0.0
    recall = (true_positives / target_positives)
    specificity = true_negatives / target_negatives
    bacc = (specificity + recall) / 2
    f1 = None if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)

    return (precision, recall, f1, bacc)