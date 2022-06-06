import numpy as np
import matplotlib.pylab as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from plotka import plot_decision_regions
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier


def main():
    iris = datasets.load_iris()
    X = iris.data[:, [2, 3]]
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

    csf = tree.DecisionTreeClassifier(criterion="entropy", max_depth=5)
    csf = tree.DecisionTreeClassifier(criterion="gini", max_depth=5)
    csf = RandomForestClassifier(n_estimators=10)
    csf.fit(X_train, y_train)

    plot_decision_regions(X=X_train, y=y_train, classifier=csf)
    plt.legend(loc='upper left')

if __name__ == '__main__':
    main()
