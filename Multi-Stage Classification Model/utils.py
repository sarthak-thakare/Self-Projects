# Updated utils.py (modular and general for any dataset)
import numpy as np
from sklearn.svm import SVC

class ClassificationPipeline:
    def __init__(self, svm_target=0, input_size=4, output_size=3):
        self.svm_target = svm_target
        self.input_size = input_size
        self.output_size = output_size
        self.h1 = 4
        self.w1 = np.random.rand(self.input_size, self.h1)
        self.b1 = np.random.rand(1, self.h1)
        self.w2 = np.random.rand(self.h1, self.output_size)
        self.b2 = np.random.rand(1, self.output_size)
        self.lr = 0.01
        self.epochs = 10000
        self.svm = SVC(kernel='linear')

    def fitSVM(self, X, y):
        binary_y = np.where(y == self.svm_target, 1, 0)
        self.svm.fit(X, binary_y)

    def probsSVM(self, X):
        scores = self.svm.decision_function(X)
        return 1 / (1 + np.exp(-scores))  # sigmoid

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def grad_sigmoid(self, a):
        return a * (1 - a)

    def loss(self, y, y_hat):
        return -np.sum(y * np.log(y_hat + 1e-9))

    def one_hot_y(self, y):
        m = y.shape[0]
        ans = np.zeros((m, self.output_size))
        for i in range(m):
            ans[i][y[i]] = 1
        return ans

    def forward(self, X):
        z1 = X @ self.w1 + self.b1
        a1 = self.sigmoid(z1)
        z2 = a1 @ self.w2 + self.b2
        a2 = self.sigmoid(z2)
        return [z1, a1, z2, a2]

    def backward(self, X, y, output):
        z1, a1, z2, a2 = output
        y_one_hot = self.one_hot_y(y)
        dz2 = (a2 - y_one_hot) * self.grad_sigmoid(a2)
        dw2 = a1.T @ dz2
        db2 = np.sum(dz2, axis=0)
        dz1 = (dz2 @ self.w2.T) * self.grad_sigmoid(a1)
        dw1 = X.T @ dz1
        db1 = np.sum(dz1, axis=0)

        self.w1 -= self.lr * dw1
        self.b1 -= self.lr * db1
        self.w2 -= self.lr * dw2
        self.b2 -= self.lr * db2

    def fitFFNN(self, X, y):
        for _ in range(self.epochs):
            output = self.forward(X)
            self.backward(X, y, output)

    def predict(self, X):
        t1 = self.probsSVM(X)  # target class confidence
        z1, a1, z2, a2 = self.forward(X)
        predictions = []
        for i in range(len(X)):
            scores = np.copy(a2[i])
            scores[self.svm_target] = t1[i]
            predictions.append(np.argmax(scores))
        return np.array(predictions)

    def fit(self, X, y):
        self.fitSVM(X, y)
        self.fitFFNN(X, y)


class StandardNNClassifier:
    def __init__(self, input_size=4, output_size=3):
        self.input_size = input_size
        self.output_size = output_size
        self.h1 = 4
        self.w1 = np.random.rand(input_size, self.h1)
        self.b1 = np.random.rand(1, self.h1)
        self.w2 = np.random.rand(self.h1, output_size)
        self.b2 = np.random.rand(1, output_size)
        self.lr = 0.01
        self.epochs = 10000

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def grad_sigmoid(self, a):
        return a * (1 - a)

    def loss(self, y, y_hat):
        return -np.sum(y * np.log(y_hat + 1e-9))

    def one_hot_y(self, y):
        m = y.shape[0]
        ans = np.zeros((m, self.output_size))
        for i in range(m):
            ans[i][y[i]] = 1
        return ans

    def forward(self, X):
        z1 = X @ self.w1 + self.b1
        a1 = self.sigmoid(z1)
        z2 = a1 @ self.w2 + self.b2
        a2 = self.sigmoid(z2)
        return [z1, a1, z2, a2]

    def backward(self, X, y, output):
        z1, a1, z2, a2 = output
        y_one_hot = self.one_hot_y(y)
        dz2 = (a2 - y_one_hot) * self.grad_sigmoid(a2)
        dw2 = a1.T @ dz2
        db2 = np.sum(dz2, axis=0)
        dz1 = (dz2 @ self.w2.T) * self.grad_sigmoid(a1)
        dw1 = X.T @ dz1
        db1 = np.sum(dz1, axis=0)

        self.w1 -= self.lr * dw1
        self.b1 -= self.lr * db1
        self.w2 -= self.lr * dw2
        self.b2 -= self.lr * db2

    def fit(self, X, y):
        for _ in range(self.epochs):
            output = self.forward(X)
            self.backward(X, y, output)

    def predict(self, X):
        _, _, _, a2 = self.forward(X)
        return np.argmax(a2, axis=1)
