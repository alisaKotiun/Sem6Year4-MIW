import numpy as np
import matplotlib.pylab as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from plotka import plot_decision_regions


class LogisticRegressionGD(object):
    def __init__(self, eta=0.05, n_iter=100, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        self.cost_ = []

        for i in range(self.n_iter):
            net_input = self.net_input(X)
            output = self.activation(net_input)
            errors = (y - output)
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            cost = (-y.dot(np.log(output)) - ((1 - y).dot(np.log(1 - output))))
            self.cost_.append(cost)
    
        return self

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, z):
        return 1. / (1. + np.exp(-np.clip(z, -250, 250)))

    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, 0)
    

class Classifier:
    def __init__(self, rl1, rl2):
        self.rl1 = rl1
        self.rl2 = rl2

    def predict(self, x):
        return np.where(self.rl1.predict(x) == 1, 0, np.where(self.rl2.predict(x) == 1, 2, 1))

def main():
    iris = datasets.load_iris()
    X = iris.data[:, [2, 3]]
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

    y_train0 = np.copy(y_train)
    y_train2 = np.copy(y_train)
    
    y_train0[y_train == 0 ] = 1
    y_train0[y_train != 0] = 0
    
    y_train2[y_train == 2] = 1
    y_train2[y_train != 2] = 0

    lrgd0 = LogisticRegressionGD(eta=0.05, n_iter=1000, random_state=1)
    lrgd2 = LogisticRegressionGD(eta=0.05, n_iter=1000, random_state=1)

    lrgd0.fit(X_train, y_train0)
    lrgd2.fit(X_train, y_train2)

    csf = Classifier(lrgd0, lrgd2) 

    plot_decision_regions(X=X_train, y=y_train, classifier=csf)
    
    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$x_2$')
    plt.legend(loc='upper left')
    plt.show()


if __name__ == '__main__':
    main()
