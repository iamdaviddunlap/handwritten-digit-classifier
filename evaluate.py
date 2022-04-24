import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

from load_data import LABEL_TO_NAME, DATASETS

# allowing for full output of DataFrame
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)


def scores(y_true, y_pred, labels):
    df_confusion = pd.crosstab(pd.Series(y_true), pd.Series(y_pred),
                               rownames=['Actual'], colnames=['Predicted'], margins=True)
    print(df_confusion)

    precision, recall, fscore, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels)
    for i, label in enumerate(labels):
        print(label)
        print(f'Precision: {round(precision[i], 2)}')
        print(f'Recall: {round(recall[i], 2)}')
        print(f'F Score: {round(fscore[i], 2)}')
        print()

    print(f'Average Precision: {round(np.mean(precision), 2)}')
    print(f'Average Recall: {round(np.mean(recall), 2)}')
    print(f'Average F1: {round(np.mean(fscore), 2)}')
    print()
    print()


def evaluate(y_true, y_pred):
    # confusion matrices

    print('All')
    y_true_label = [LABEL_TO_NAME[tuple(label)] for label in y_true]
    y_pred_label = [LABEL_TO_NAME[tuple(label)] for label in y_pred]
    labels = list(LABEL_TO_NAME.values())
    scores(y_true_label, y_pred_label, labels)

    print('Language')
    y_true_language = [label.split()[0] for label in y_true_label]
    y_pred_language = [label.split()[0] for label in y_pred_label]
    labels = DATASETS
    scores(y_true_language, y_pred_language, labels)
