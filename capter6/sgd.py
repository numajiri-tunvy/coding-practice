"""
SGD: Stochastic Gradient Descent
"""

import random
import numpy as np

"""
シグモイド関数
"""
def sigmoid(a):
    return 1.0 / (1.0 + np.exp(-a))



def sgd(x,y):
    [N, D] = x.shape
    c = 0.25
    w = np.random.random(D)
    for t in range(1, 100):
        eta = 0.5/np.sqrt(t)
        y_pred = sigmoid(w @ x.T).reshape(-1)
        err = y - y_pred
        w += eta * (np.dot(err, x) + 2*c*w)
    return w

    
def main():
    x = np.array([
        [0, 0, 1],
        [0, 1, 1],
        [1, 0, 1],
        [1, 1, 1],
    ])
    y = np.array([1, 0, 1, 0])
    w = np.array([0.0, 0.0, 0.0])
    b = 0.0
    # データ点表示
    [N, D] = x.shape
    print("x", "      |", "y")
    for i in range(N):
        print(x[i], "|", y[i])
    #
    w = sgd(x, y)
    print(f"w={w}")

    y_pred = sigmoid(w @ x.T).reshape(-1)
    print(f"y_pred={y_pred}")

if __name__ == "__main__":
    main()