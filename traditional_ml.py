import numpy as np
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
from time import time
import random

from evaluate import evaluate
from load_data import load_data

(X_train, y_train), (X_test, y_test) = load_data(flatten=True)

# build and train the classifier
# clf = MultiOutputClassifier(KNeighborsClassifier()).fit(X_train, y_train)
clf = MultiOutputClassifier(
    MLPClassifier(
        solver='adam',
        alpha=1e-4,
        hidden_layer_sizes=(10, 5),
        random_state=1,
        verbose=10,
        max_iter=40)).fit(X_train, y_train)

# predict
print('Starting prediction')
start = time()
y_pred = clf.predict(X_test)
print(f'Finished prediction in {time() - start}s')
y_true = y_test

evaluate(y_true, y_test)
