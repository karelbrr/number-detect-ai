import os

import numpy as np
import csv


def check_if_model_saved(filepath):
    return os.path.exists(filepath)


def load_numpy_model(filepath):
    model = np.load(filepath)
    return model["W1"], model["b1"], model["W2"], model["b2"]


def check_if_model_loaded():
    if not check_if_model_saved('my_trained_data.npz'):
        X_train, Y_train, Y_raw = load_data('mnist_train.csv')
        print("Train data loaded")
        return X_train, Y_train, Y_raw
    else:
        W1, b1, W2, b2 = load_numpy_model('my_trained_data.npz')
        print("Numpy data loaded")
        return W1, b1, W2, b2


def one_hot_encoding(Y):
    representation_nut = np.eye(10)
    return representation_nut[Y]


def derivace_ReLU(Z):
    return Z > 0


def softmax(x):
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))

    return e_x / e_x.sum(axis=1, keepdims=True)


def ReLU(Z):
    return np.maximum(0, Z)


def load_data(filepath):
    data = []

    with open(filepath, 'r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)
        for row in csv_reader:
            data.append([int(x) for x in row])

    data = np.array(data)
    np.random.shuffle(data)

    Y_raw = data[:, 0]
    X = data[:, 1:]

    X = X / 255.0

    Y_train = one_hot_encoding(Y_raw)

    return X, Y_train, Y_raw


def init_params():
    np.random.seed(42)

    W1 = np.random.randn(784, 64) * 0.1
    b1 = np.zeros((1, 64))

    W2 = np.random.randn(64, 10) * 0.1
    b2 = np.zeros((1, 10))

    return W1, b1, W2, b2


def forward_prop(W1, b1, W2, b2, X_train):
    Z1 = np.dot(X_train, W1) + b1

    A1 = ReLU(Z1)

    Z2 = np.dot(A1, W2) + b2

    A2 = softmax(Z2)

    return Z1, A1, Z2, A2


def backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y):
    m = Y.shape[0]

    dZ2 = A2 - Y
    dW2 = 1 / m * np.dot(A1.T, dZ2)
    db2 = 1 / m * np.sum(dZ2, axis=0, keepdims=True)

    dZ1 = np.dot(dZ2, W2.T) * derivace_ReLU(Z1)
    dW1 = 1 / m * np.dot(X.T, dZ1)
    db1 = 1 / m * np.sum(dZ1, axis=0, keepdims=True)

    return dW1, db1, dW2, db2


alpha = 0.1


def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1
    W2 = W2 - alpha * dW2
    b2 = b2 - alpha * db2
    return W1, b1, W2, b2


def get_accuracy(prediction_A2, real_Y):
    best_estimate = np.argmax(prediction_A2, axis=1)
    return np.sum(best_estimate == real_Y) / real_Y.size


def gradient_descent(X, Y_encoded, Y_raw, alpha, iterations):
    W1, b1, W2, b2 = init_params()

    for i in range(iterations):

        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)

        dW1, db1, dW2, db2 = backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y_encoded)

        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)

        if i % 10 == 0:
            print(f"Epocha: {i}")
            presnost = get_accuracy(A2, Y_raw)
            print(f"Přesnost: {presnost * 100:.2f}%")

    return W1, b1, W2, b2


file = 'my_trained_data.npz'

if os.path.exists(file):
    print("Načítám uložený model...")
    nacteny_model = np.load(file)
    W1_final = nacteny_model['W1']
    b1_final = nacteny_model['b1']
    W2_final = nacteny_model['W2']
    b2_final = nacteny_model['b2']

else:
    print("Model nenalezen. Začínám načítat data a trénovat novou síť...")
    X_train, Y_train_encoded, Y_train_raw = load_data('mnist_train_copy.csv')

    W1_final, b1_final, W2_final, b2_final = gradient_descent(X_train, Y_train_encoded, Y_train_raw, alpha=0.1,
                                                              iterations=500)

    print("Ukládám natrénovaný model...")
    np.savez(file, W1=W1_final, b1=b1_final, W2=W2_final, b2=b2_final)
    print("Model bezpečně uložen!")
