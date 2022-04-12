import numpy as np
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix

from load_data import load_mnist, load_kannada

mnist = load_mnist()
kannada = load_kannada()

(total_x_train, total_y_train), (total_x_test, total_y_test) = mnist.get_data()
for (x_train, y_train), (x_test, y_test) in [kannada.get_data()]:
    total_x_train = np.concatenate((total_x_train, x_train))
    total_y_train = np.concatenate((total_y_train, y_train))
    total_x_test = np.concatenate((total_x_test, x_test))
    total_y_test = np.concatenate((total_y_test, y_test))

total_x_train, total_y_train = shuffle(total_x_train, total_y_train)
total_x_test, total_y_test = shuffle(total_x_test, total_y_test)

clf = MultiOutputClassifier(KNeighborsClassifier()).fit(total_x_train, total_y_train)
y_pred = clf.predict(total_x_test[:1000])
y_true = total_y_test[:1000]

print('Language')
print(confusion_matrix(y_true[:, 0], y_pred[:, 0]))
print()
print('Numeral')
print(confusion_matrix(y_true[:, 1], y_pred[:, 1]))
