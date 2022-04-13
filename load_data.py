import numpy as np
import os
import cv2

ARABIC = 'arabic'
KANNADA = 'kannada'
ODIA = 'odia'
DATASETS = [ARABIC, KANNADA, ODIA]

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
ARABIC_PATH = os.path.join(DATA_DIR, f'{ARABIC}/mnist.npz')
KANNADA_DIR = os.path.join(DATA_DIR, KANNADA)
ODIA_DIR = os.path.join(DATA_DIR, f'{ODIA}/images')


class Dataset:

    def __init__(self, X_train, y_train, X_test, y_test, language):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.language = language

        self.X_train = self._equalize(self.X_train)
        self.X_test = self._equalize(self.X_test)

        self.y_train = self._add_language_to_label(y_train)
        self.y_test = self._add_language_to_label(y_test)

    def _equalize(self, x):
        return np.array([cv2.equalizeHist(image) for image in x])

    @staticmethod
    def _flatten(x):
        return x.reshape((len(x), -1))

    def _add_language_to_label(self, y):
        language_num = DATASETS.index(self.language)
        y = np.vstack((np.full(len(y), language_num), y.T)).T
        return y

    def get_X_train(self, flatten=False):
        if flatten:
            return self._flatten(self.X_train)
        return self.X_train

    def get_y_train(self):
        return self.y_train

    def get_X_test(self, flatten=False):
        if flatten:
            return self._flatten(self.X_test)
        return self.X_test

    def get_y_test(self):
        return self.y_test


def load_arabic():
    with np.load(ARABIC_PATH) as f:
        X_train, y_train = f['x_train'], f['y_train']
        X_test, y_test = f['x_test'], f['y_test']

        return Dataset(X_train, y_train, X_test, y_test, ARABIC)


def load_kannada():
    X_train = np.load(os.path.join(KANNADA_DIR, 'X_kannada_MNIST_train.npz'))['arr_0']
    X_test = np.load(os.path.join(KANNADA_DIR, 'X_kannada_MNIST_test.npz'))['arr_0']
    y_train = np.load(os.path.join(KANNADA_DIR, 'y_kannada_MNIST_train.npz'))['arr_0']
    y_test = np.load(os.path.join(KANNADA_DIR, 'y_kannada_MNIST_test.npz'))['arr_0']

    return Dataset(X_train, y_train, X_test, y_test, KANNADA)


def load_odia():
    X_train = np.load(os.path.join(ODIA_DIR, 'X_train.npy'))
    X_test = np.load(os.path.join(ODIA_DIR, 'X_test.npy'))
    y_train = np.load(os.path.join(ODIA_DIR, 'y_train.npy'))
    y_test = np.load(os.path.join(ODIA_DIR, 'y_test.npy'))

    return Dataset(X_train, y_train, X_test, y_test, ODIA)


def load_datasets():
    return {
        ARABIC: load_arabic(),
        KANNADA: load_kannada(),
        ODIA: load_odia()
    }


if __name__ == '__main__':
    mnist = load_arabic()
    kannada = load_kannada()
    odia = load_odia()
    x=1