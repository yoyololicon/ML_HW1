import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == '__main__':
    test_data = pd.read_csv('data/4_test.csv')
    train_data = pd.read_csv('data/4_train.csv')

    train_t = np.array(train_data['t'].values)
    test_t = np.array(test_data['t'].values)
    train_x = np.array(train_data['x'].values)
    test_x = np.array(test_data['x'].values)

    train_elist = []
    test_elist = []

    for order in range(0, 10):
        Xm = np.zeros((len(train_x), order+1))
        Xm[:, 0] = 1
        for i in range(1, order+1):
            Xm[:, i] = train_x**i

        Xt = np.transpose(Xm)
        w = np.dot(np.dot(np.linalg.inv(np.dot(Xt, Xm)), Xt), train_t)

        w = np.flip(w, 0)
        p = np.poly1d(w)

        #print w, np.polyfit(train_x, train_t, order)

        train_y = p(train_x)
        test_y = p(test_x)

        train_elist.append(np.sqrt(np.sum((train_y - train_t) ** 2) / len(train_t)))
        test_elist.append(np.sqrt(np.sum((test_y - test_t) ** 2) / len(test_t)))

    plt.plot(train_elist, marker='o')
    plt.plot(test_elist, 'r-', marker='o')
    plt.xlim(-1, 10)
    plt.xlabel('M')
    plt.ylabel('RMS Error')
    plt.show()