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

KNN = 'knn'
RANDOM_FOREST = 'random_forest'
MLP = 'mlp'

CLF = KNN
TRAIN = True


def main():
    if CLF == KNN:
        joblib_name = 'knn.joblib'
    elif CLF == RANDOM_FOREST:
        joblib_name = 'random_forest.joblib'
    elif CLF == MLP:
        joblib_name = 'mlp.joblib'

    JOBLIB_PATH = os.path.join(MODELS_DIR, joblib_name)

    # load the data
    (X_train, y_train), (X_test, y_test) = load_data(flatten=True)

    if TRAIN:
        # build and train the classifier
        print('Training...')

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

        clf.fit(X_train, y_train)
        dump(clf, JOBLIB_PATH)
    else:
        clf = load(JOBLIB_PATH)

    # predict
    print('Predicting...')
    start = time()
    y_pred = clf.predict(X_test)
    print(f'Finished prediction in {round(time() - start, 2)}s')
    y_true = y_test

    evaluate(y_true, y_pred)


if __name__ == '__main__':
    main()
