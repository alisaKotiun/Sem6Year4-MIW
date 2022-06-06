import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def main():
    a = np.loadtxt('as7\dane6.txt')

    x = a[:,[0]]
    y = a[:,[1]]

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

    P = np.vstack(X_train).T
    T = np.power(P, 2) + 1 * (np.random.rand(P.shape[0], P.shape[1]) - 0.5)

    S1 = 100
    W1 = np.random.rand(S1, 1) - 0.5
    B1 = np.random.rand(S1, 1) - 0.5
    W2 = np.random.rand(1, S1) - 0.5
    B2 = np.random.rand(1, 1) - 0.5
    lr = 0.001

    for i in range(1, 201):
        s = W1 @ P + B1 @ np.ones(P.shape)
        A1 = np.arctan(s)
        A2 = W2 @ A1 + B2

        E2 = T - A2
        E1 = W2.T @ E2

        dW2 = lr * E2 @ A1.T
        dB2 = lr * E2 @ np.ones(E2.shape).T
        dW1 = lr * (1 / (1 + np.power(s, 2))) * E1 @ P.T
        dB1 = lr * (1 / (1 + np.power(s, 2))) * E1 @ np.ones(P.shape).T

        W2 = W2 + dW2
        B2 = B2 + dB2
        W1 = W1 + dW1
        B1 = B1 + dB1
            
    plt.plot(P, T, 'ro')
    plt.plot(P, A2, 'b-')
    plt.show()

    e_train = P - A2
    print('acc:' + str(e_train@e_train.T/len(e_train)))

if __name__ == '__main__':
    main()