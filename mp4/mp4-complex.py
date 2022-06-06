import numpy as np
import matplotlib.pylab as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split

def main():
    a = np.loadtxt('as6\dane6.txt')

    x = a[:,[0]]
    y = a[:,[1]]

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3)


    c = np.hstack([X_train*X_train, X_train, np.ones(X_train.shape)]) 
    v = np.linalg.pinv(c)@y_train

    e_train = y_train - (v[0]*X_train*X_train + v[1]*X_train + v[2])
    e_test = y_test - (v[0]*X_test*X_test + v[1]*X_test + v[2]) 
    
    print('y = ax^2 + bx + c')
    print('train: ' + str(e_train.T@e_train/len(e_train)))
    print('test: ' + str(e_test.T@e_test/len(e_test)))


    plt.plot(X_test, y_test, 'ro')
    plt.plot(X_test, v[0]*X_test*X_test + v[1]*X_test + v[2], 'b*')
    plt.show()

if __name__ == '__main__':
    main()

    #y=ax+b
    #y=ax2 + bx + c

