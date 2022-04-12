import numpy as np
import os
import cv2

ARABIC = 'arabic'
KANNADA = 'kannada'
ODIA = 'odia'
DATASETS = [ARABIC, KANNADA, ODIA]

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
MNIST_PATH = os.path.join(DATA_DIR, f'{ARABIC}/mnist.npz')
KANNADA_DIR = os.path.join(DATA_DIR, KANNADA)


class Dataset:

    def __init__(self, x_train, y_train, x_test, y_test, language):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.language = language

        self.x_train = self._equalize(self.x_train)
        self.x_test = self._equalize(self.x_test)

        self.x_train = self._flatten(self.x_train)
        self.x_test = self._flatten(self.x_test)

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

    def get_data(self):
        return (self.x_train, self.y_train), (self.x_test, self.y_test)



# def add_language_to_label(y, language):
#     language_num = DATASETS.index(language)
#     y = np.vstack((np.full(len(y), language_num), y.T)).T
#     return y


def load_mnist():
    with np.load(MNIST_PATH) as f:
        x_train, y_train = f['x_train'], f['y_train']
        x_test, y_test = f['x_test'], f['y_test']

        return Dataset(x_train, y_train, x_test, y_test, ARABIC)

        # x_train.reshape((len(x_train), -1))
        # x_test.reshape((len(x_test), -1))
        #
        # y_train = add_language_to_label(y_train, MNIST)
        # y_test = add_language_to_label(y_test, MNIST)
        #
        # return (x_train, y_train), (x_test, y_test)


def load_kannada():
    x_train = np.load(os.path.join(KANNADA_DIR, 'X_kannada_MNIST_train.npz'))['arr_0']
    x_test = np.load(os.path.join(KANNADA_DIR, 'X_kannada_MNIST_test.npz'))['arr_0']
    y_train = np.load(os.path.join(KANNADA_DIR, 'y_kannada_MNIST_train.npz'))['arr_0']
    y_test = np.load(os.path.join(KANNADA_DIR, 'y_kannada_MNIST_test.npz'))['arr_0']

    return Dataset(x_train, y_train, x_test, y_test, KANNADA)

    # x_train.reshape((len(x_train), -1))
    # x_test.reshape((len(x_test), -1))
    #
    # y_train = add_language_to_label(y_train, KANNADA)
    # y_test = add_language_to_label(y_test, KANNADA)
    #
    # return (x_train, y_train), (x_test, y_test)


if __name__ == '__main__':
    mnist = load_mnist()
    kannada = load_kannada()
    x=1