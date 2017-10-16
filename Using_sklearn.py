import numpy as np
import matplotlib.pyplot as plt
import scipy.io as io
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

if __name__ == '__main__':
    data = io.loadmat('data/5_X.mat')['X']
    target = io.loadmat('data/5_T.mat')['T']

    train_x = np.concatenate((data[:40], data[50:90], data[100:140]))
    test_x = np.concatenate((data[40:50], data[90:100], data[140:150]))
    train_t = np.concatenate((target[:40], target[50:90], target[100:140]))
    test_t = np.concatenate((target[40:50], target[90:100], target[140:150]))

    poly = PolynomialFeatures(degree=2)
    x_trans = poly.fit_transform(2)
    test_trans = poly.fit_transform(test_x)

    #lg = LinearRegression()
    #lg.fit(x_trans, train_t)

    print data[0], x_trans[0]
