import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def multilabel_auc(y_true, y_score, classes, print_scores=True):
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    scores = []
    known_counts = []

    for i, name in enumerate(classes):
        known = y_true[:, i] != -1
        target = y_true[known, i]
        score = y_score[known, i]
        auc = roc_auc_score(target, score) if len(np.unique(target)) == 2 else np.nan
        scores.append(auc)
        known_counts.append(int(known.sum()))

    result = pd.DataFrame({"class": classes, "auc": scores, "known": known_counts})
    mean = float(np.nanmean(scores)) if np.any(~np.isnan(scores)) else np.nan

    if print_scores:
        print(result.to_string(index=False, formatters={"auc": lambda x: f"{x:.4f}"}))
        print("Mean                 {:.4f}".format(mean))

    return mean, np.asarray(scores), result
