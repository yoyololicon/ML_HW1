import numpy as np
import matplotlib.pyplot as plt
import scipy.io as io
from itertools import combinations_with_replacement

def getCoef(data, target):
    Xt = np.transpose(data)
    w = np.dot(np.dot(np.linalg.inv(np.dot(Xt, data)), Xt), target)
    return w

def transform(data, dim, order):
    trans_data = np.full((len(data), 1), 1)
    for i in range(order):
        for comb in list(combinations_with_replacement(range(dim), i+1)):
            p = np.expand_dims(np.prod(data[:, comb], axis=1), axis=1)
            trans_data = np.concatenate((trans_data, p), axis=1)

    return trans_data

def RMSE(y, t):
    return np.sqrt(np.mean((y - t) ** 2))

if __name__ == '__main__':
    data = io.loadmat('data/5_X.mat')['X']
    target = io.loadmat('data/5_T.mat')['T']

    train_x = np.concatenate((data[:40], data[50:90], data[100:140]))
    test_x = np.concatenate((data[40:50], data[90:100], data[140:150]))
    train_t = np.concatenate((target[:40], target[50:90], target[100:140]))
    test_t = np.concatenate((target[40:50], target[90:100], target[140:150]))

    #order = 1

    trans_train_x = transform(train_x, 4, 1)
    trans_test_x = transform(test_x, 4, 1)

    w = getCoef(trans_train_x, train_t)

    train_y1 = np.dot(trans_train_x, w)
    test_y1 = np.dot(trans_test_x, w)

    train_error1 = RMSE(train_y1, train_t)
    test_error1 = RMSE(test_y1, test_t)
    print 'For M = 1:'
    print 'Training error is', train_error1, ", testing error is", test_error1

    #order = 2

    trans_train_x = transform(train_x, 4, 2)
    trans_test_x = transform(test_x, 4, 2)

    w = getCoef(trans_train_x, train_t)

    train_y2 = np.dot(trans_train_x, w)
    test_y2 = np.dot(trans_test_x, w)

    train_error2 = RMSE(train_y2, train_t)
    test_error2 = RMSE(test_y2, test_t)
    print 'For M = 2:'
    print 'Training error is', train_error2, ", testing error is", test_error2


    plt.plot(test_y1, 'r')
    plt.plot(test_y2)
    plt.show()

