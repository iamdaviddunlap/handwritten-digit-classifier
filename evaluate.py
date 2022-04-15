import numpy as np
from sklearn.metrics import confusion_matrix


def evaluate(y_true, y_pred):
    # confusion matrices
    print('Language')
    print(confusion_matrix(y_true[:, 0], y_pred[:, 0]))
    print()
    print('Numeral')
    print(confusion_matrix(y_true[:, 1], y_pred[:, 1]))
    print()
    for numeral in range(10):
        print(numeral)
        indices = np.where((y_pred[:, 1] == numeral) | (y_true[:, 1] == numeral))
        language_pred = y_pred[indices, 0].squeeze()
        language_true = y_true[indices, 0].squeeze()
        print(confusion_matrix(language_true, language_pred))
        print()