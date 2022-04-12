import numpy as np
import os


DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
MNIST_PATH = os.path.join(DATA_DIR, 'MNIST/mnist.npz')
KANNADA_DIR = os.path.join(DATA_DIR, 'kannada')


def load_mnist():
    with np.load(MNIST_PATH) as f:
        x_train, y_train = f['x_train'], f['y_train']
        x_test, y_test = f['x_test'], f['y_test']
        return (x_train, y_train), (x_test, y_test)


def load_kannada():
    x_train = np.load(os.path.join(KANNADA_DIR, 'X_kannada_MNIST_train.npz'))['arr_0']
    x_test = np.load(os.path.join(KANNADA_DIR, 'X_kannada_MNIST_test.npz'))['arr_0']
    y_train = np.load(os.path.join(KANNADA_DIR, 'y_kannada_MNIST_train.npz'))['arr_0']
    y_test = np.load(os.path.join(KANNADA_DIR, 'y_kannada_MNIST_test.npz'))['arr_0']

    return (x_train, y_train), (x_test, y_test)


if __name__ == '__main__':
    mnist = load_mnist()
    kannada = load_kannada()
    x=1