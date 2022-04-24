"""
An implementation of various machine learning models using sci-kit learn.
"""

import os

from joblib import dump, load
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from time import time

from evaluate import evaluate
from load_data import load_data

MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')
if not os.path.exists(MODELS_DIR):
    os.makedirs(MODELS_DIR)

KNN = 'KNeighborsClassifier'
RANDOM_FOREST = 'RandomForestClassifier'
MLP = 'MLPClassifier'

# Change these as appropriate
CLF = KNN
TRAIN = True


def main():
    # path to the saved model (or model to save)
    joblib_name = f'{CLF}.joblib'
    JOBLIB_PATH = os.path.join(MODELS_DIR, joblib_name)

    # load the data
    (X_train, y_train), (X_test, y_test) = load_data(flatten=True)

    # train and eval or only eval
    if TRAIN:
        # build and train the classifier
        print(f'Training {CLF}...')

        # instantiate the classifier
        if CLF == KNN:
            clf = MultiOutputClassifier(KNeighborsClassifier())
        elif CLF == RANDOM_FOREST:
            clf = MultiOutputClassifier(RandomForestClassifier())
        elif CLF == MLP:
            clf = MultiOutputClassifier(
                MLPClassifier(
                    solver='adam',
                    alpha=1e-4,
                    hidden_layer_sizes=(10, 5),
                    random_state=1,
                    verbose=10,
                    max_iter=40))

        # train the classifier
        clf.fit(X_train, y_train)

        # save the model
        dump(clf, JOBLIB_PATH)
        print(f'Output {joblib_name}')
    else:
        # load the model
        clf = load(JOBLIB_PATH)
        print(f'Loaded {CLF}')

    # predict
    print('Predicting...')
    start = time()
    y_pred = clf.predict(X_test)
    print(f'Finished prediction in {round(time() - start, 2)}s')
    y_true = y_test

    # evaluate the model
    evaluate(y_true, y_pred)


if __name__ == '__main__':
    main()
