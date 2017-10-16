import numpy as np
import matplotlib.pyplot as plt
import scipy.io as io
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

def getCoef(data, target):
    Xt = np.transpose(data)
    w = np.dot(np.dot(np.linalg.inv(np.dot(Xt, data)), Xt), target)
    return w

if __name__ == '__main__':
    data = io.loadmat('data/5_X.mat')['X']
    target = io.loadmat('data/5_T.mat')['T']

    train_x = np.concatenate((data[:40], data[50:90], data[100:140]))
    test_x = np.concatenate((data[40:50], data[90:100], data[140:150]))
    train_t = np.concatenate((target[:40], target[50:90], target[100:140]))
    test_t = np.concatenate((target[40:50], target[90:100], target[140:150]))

    #order = 1

    trans_train_x = np.concatenate((np.full((len(train_x), 1), 1), train_x), axis=1)
    trans_test_x = np.concatenate((np.full((len(test_x), 1), 1), test_x), axis=1)

    w = getCoef(trans_train_x, train_t)

    train_y = np.dot(trans_train_x, w)
    test_y = np.dot(trans_test_x, w)

    #print test_y, test_t
    print np.sqrt(np.mean((train_y - train_t) ** 2)), np.sqrt(np.mean((test_y - test_t) ** 2))

    poly = PolynomialFeatures(degree=1)
    x_trans = poly.fit_transform(train_x)
    #test_trans = poly.fit_transform(test_x)

    lg = LinearRegression()
    lg.fit(x_trans, train_t)

    yy = lg.predict(x_trans)
    plt.plot(yy, 'r')
    plt.plot(np.round(train_y), 'b')
    plt.show()

