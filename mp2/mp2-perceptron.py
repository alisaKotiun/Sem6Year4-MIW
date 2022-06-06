import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
from plotka import plot_decision_regions

class Perceptron(object):

    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        self.w_ = np.zeros(1+ X.shape[1])
        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X,y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update *xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, -1)

class Classifier:
    def __init__(self, p1, p2):
        self.p1 = p1
        self.p2 = p2

    def predict(self, x):
        return np.where(self.p1.predict(x) == 1, 0, np.where(self.p2.predict(x) == 1, 2, 1))

def main():

    iris = datasets.load_iris()
    X = iris.data[:, [2, 3]]
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

    y_train0 = np.copy(y_train)
    y_train2 = np.copy(y_train)

    y_train0[y_train == 0] = 1
    y_train0[y_train != 0] = -1

    y_train2[y_train == 2] = 1
    y_train2[y_train != 2] = -1
       
    p1 = Perceptron(eta=0.01, n_iter=1000)
    p2 = Perceptron(eta=0.01, n_iter=1000)

    p1.fit(X_train, y_train0)
    p2.fit(X_train, y_train2)

    csf = Classifier(p1, p2) 

    plot_decision_regions(X=X_train, y=y_train, classifier=csf)
    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$x_2$')
    plt.legend(loc='upper left')
    plt.show()
    

if __name__ == '__main__':
    main()
