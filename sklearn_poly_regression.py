import numpy as np
import matplotlib.pyplot as plt
import scipy.io as io
import poly_regression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

if __name__ == '__main__':
    data = io.loadmat('data/5_X.mat')['X']
    target = io.loadmat('data/5_T.mat')['T']

    train_x = np.concatenate((data[:40], data[50:90], data[100:140]))
    test_x = np.concatenate((data[40:50], data[90:100], data[140:150]))
    train_t = np.concatenate((target[:40], target[50:90], target[100:140]))
    test_t = np.concatenate((target[40:50], target[90:100], target[140:150]))

    #order = 1

    poly1 = PolynomialFeatures(degree=1)

    trans_train_x = poly1.fit_transform(train_x)
    trans_test_x = poly1.fit_transform(test_x)

    lg1 = LinearRegression()
    lg1.fit(trans_train_x, train_t)

    train_y1 = lg1.predict(trans_train_x)
    test_y1 = lg1.predict(trans_test_x)

    train_error1 = poly_regression.RMSE(train_y1, train_t)
    test_error1 = poly_regression.RMSE(test_y1, test_t)
    print 'For M = 1:'
    print 'Training error is', train_error1, ", testing error is", test_error1

    #order = 2

    poly2 = PolynomialFeatures(degree=2)

    trans_train_x = poly2.fit_transform(train_x)
    trans_test_x = poly2.fit_transform(test_x)

    lg2 = LinearRegression()
    lg2.fit(trans_train_x, train_t)

    train_y2 = lg2.predict(trans_train_x)
    test_y2 = lg2.predict(trans_test_x)

    train_error2 = poly_regression.RMSE(train_y2, train_t)
    test_error2 = poly_regression.RMSE(test_y2, test_t)
    print 'For M = 2:'
    print 'Training error is', train_error2, ", testing error is", test_error2

    plt.plot(test_y1, 'r')
    plt.plot(test_y2)
    plt.show()

