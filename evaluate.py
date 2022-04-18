import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from load_data import LABEL_TO_NAME

# allowing for full output of DataFrame
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)


def evaluate(y_true, y_pred):
    # confusion matrices

    print('All')
    y_true_label = [LABEL_TO_NAME[tuple(label)] for label in y_true]
    y_pred_label = [LABEL_TO_NAME[tuple(label)] for label in y_pred]

    df_confusion = pd.crosstab(pd.Series(y_true_label), pd.Series(y_pred_label),
                               rownames=['Actual'], colnames=['Predicted'], margins=True)
    print(df_confusion)

    print('Language')
    print(confusion_matrix(y_true[:, 0], y_pred[:, 0]))
    print()
    print('Numeral')
    print(confusion_matrix(y_true[:, 1], y_pred[:, 1]))
    print()
    for numeral in range(10):
        print(numeral)
        indices = np.where((y_pred[:, 1] == numeral) & (y_true[:, 1] == numeral))
        language_pred = y_pred[indices, 0].squeeze()
        language_true = y_true[indices, 0].squeeze()
        print(confusion_matrix(language_true, language_pred))
        print()