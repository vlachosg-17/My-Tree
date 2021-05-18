import numpy as np
import pandas as pd
# from sklearn.metrics import confusion_matrix

def counts(x):
    u, c = np.unique(x, return_counts=True)
    return dict(zip(u, c))

def myconfusion_matrix(y_true, y_pred):
    if len(np.unique(y_true)) >= len(np.unique(y_pred)):
        categ = np.unique(y_true)
    else:
        categ = np.unique(y_pred)
    cm = np.zeros((len(categ), len(categ)))
    for pred, label in zip(y_pred, y_true):
        for i, c1 in enumerate(categ):
            if pred == c1 and label == c1:
                cm[i, i] += 1
                break
            else:
                for j, c2 in enumerate(categ):
                    if pred == c2 and label == c1:
                        cm[i, j] += 1
                        break 
    cm = cm.astype(int)
    return cm

def accuracy_score(cm):
    return cm.trace() / cm.sum()

def precision_class(cm):
    rows, _ = cm.shape
    sum_precisions = 0
    for label in range(rows):
        col = cm[:, label]
        sum_precisions += cm[label, label] / sum(col)
    return sum_precisions / rows

def recall_class(cm):
    _, columns = cm.shape
    sum_recalls = 0
    for label in range(columns):
        row = cm[label, :]
        sum_recalls += cm[label, label] / sum(row)
    return sum_recalls / columns

def metrics(y_pred, y, model=None):
    if model is not None:
        cv, train = None, None
        if model.type == "randomforest":
            if model.cv_error is not None: 
                cv = round(model.cv_error, 3)
            if model.train_error is not None:
                train = round(model.train_error, 3)
        # if model.type == "randomforest":

    else:
        cv, train = None, None
    cm = myconfusion_matrix(y, y_pred)
    accuracy = round(accuracy_score(cm), 3)
    precision = round(precision_class(cm), 3)
    recall = round(recall_class(cm), 3)
    F1 = 2*recall*precision / (recall+precision)
    cm_ = pd.DataFrame(cm, index = [f"true_lab{i}" for i in range(cm.shape[0])], \
            columns=[f"pred_lab{i}" for i in range(cm.shape[0])])
    mtrcs = pd.DataFrame({
        "oob_error": None,
        "train_error": train,
        "cv_error": cv,
        "accuracy": accuracy,
        "clf_error": accuracy,
        "precision": precision,
        "recall": recall,
        "Fmeasure": F1
        }, index=["metrics"])
    return {'cm': cm_, "measures": mtrcs}

def print_metrics(m):
    types=[float, np.float16, np.float32, np.float64,\
           int, np.int8, np.int16, np.int32, type(None)]
    n=-1
    for key, item in m.items():
        if n < len(key):
            n = len(key)
    for _, (key, item) in enumerate(m.items()):
        print(" "*n)
        if type(item) in types:
            print(key+": "+str(item))
        else:     
            print(key+": ")
            print(item)
        