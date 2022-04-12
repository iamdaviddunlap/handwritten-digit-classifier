import numpy as np
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
from time import time
import random

from load_data import load_mnist, load_kannada

mnist_dataset = load_mnist()
kannada_dataset = load_kannada()

(total_x_train, total_y_train), (total_x_test, total_y_test) = mnist_dataset.get_data()
for (x_train, y_train), (x_test, y_test) in [kannada_dataset.get_data()]:
    total_x_train = np.concatenate((total_x_train, x_train))
    total_y_train = np.concatenate((total_y_train, y_train))
    total_x_test = np.concatenate((total_x_test, x_test))
    total_y_test = np.concatenate((total_y_test, y_test))

total_x_train, total_y_train = shuffle(total_x_train, total_y_train)
total_x_test, total_y_test = shuffle(total_x_test, total_y_test)

# clf = MultiOutputClassifier(KNeighborsClassifier()).fit(total_x_train, total_y_train)
clf = MultiOutputClassifier(MLPClassifier(solver='adam', alpha=1e-4, hidden_layer_sizes=(10, 5),
                                          random_state=1, verbose=10, max_iter=40))\
    .fit(total_x_train, total_y_train)

print('Starting prediction')
start = time()
y_pred = clf.predict(total_x_test[:500])
print(f'Finished prediction in {time() - start}s')
y_true = total_y_test[:500]

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

