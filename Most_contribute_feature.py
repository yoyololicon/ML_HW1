import numpy as np
import scipy.io as io
import Poly_regress

feature = ['sepal length', 'sepal width', 'petal length', 'petal width']

if __name__ == '__main__':
    data = io.loadmat('data/5_X.mat')['X']
    target = io.loadmat('data/5_T.mat')['T']

    train_x = np.concatenate((data[:40], data[50:90], data[100:140]))
    train_t = np.concatenate((target[:40], target[50:90], target[100:140]))

    error_list = []
    for i in range(4):
        trans_train_x = Poly_regress.transform(np.delete(train_x, i, 1), 3, 2)

        w = Poly_regress.getCoef(trans_train_x, train_t)

        train_y = np.dot(trans_train_x, w)

        error_list.append(Poly_regress.RMSE(train_y, train_t))
        print 'The RMS error after remove feature <<', feature[i], '>> is', error_list[len(error_list)-1]

    print 'The most contributive attribute is', feature[error_list.index(max(error_list))]

