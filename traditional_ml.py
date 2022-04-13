import numpy as np
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
from time import time
import random

from load_data import load_datasets

# load the datasets
datasets = load_datasets()

# get the train and test data
X_train = np.concatenate([dataset.get_X_train(flatten=True) for dataset in datasets.values()])
y_train = np.concatenate([dataset.get_y_train() for dataset in datasets.values()])
X_test = np.concatenate([dataset.get_X_test(flatten=True) for dataset in datasets.values()])
y_test = np.concatenate([dataset.get_y_test() for dataset in datasets.values()])

# shuffle the data
X_train, y_train = shuffle(X_train, y_train)
X_test, y_test = shuffle(X_test, y_test)

# build and train the classifier
clf = MultiOutputClassifier(KNeighborsClassifier()).fit(X_train, y_train)
# clf = MultiOutputClassifier(
#     MLPClassifier(
#         solver='adam',
#         alpha=1e-4,
#         hidden_layer_sizes=(10, 5),
#         random_state=1,
#         verbose=10,
#         max_iter=40)).fit(X_train, y_train)

# predict
print('Starting prediction')
start = time()
y_pred = clf.predict(X_test)
print(f'Finished prediction in {time() - start}s')
y_true = y_test

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

