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

    order = 9
    lnl = np.linspace(-20, 0, 100)

    for l in np.nditer(lnl):
        Xm = np.zeros((len(train_x), order+1))
        Xm[:, 0] = 1
        for i in range(1, order+1):
            Xm[:, i] = train_x**i

        ld = np.exp(l)
        Xt = np.transpose(Xm)
        w = np.dot(np.dot(np.linalg.inv(ld*np.identity(order+1) + np.dot(Xt, Xm)), Xt), train_t)

        w = np.flip(w, 0)
        p = np.poly1d(w)

        #print w, np.polyfit(train_x, train_t, order)

        train_y = p(train_x)
        test_y = p(test_x)

        train_elist.append(np.sqrt(np.sum((train_y - train_t) ** 2) / len(train_t)))
        test_elist.append(np.sqrt(np.sum((test_y - test_t) ** 2) / len(test_t)))

    x1 = lnl.tolist()
    plt.plot(x1, train_elist)
    plt.plot(x1, test_elist, 'r-')
    plt.xlim(-21, 1)

    plt.xlabel('ln lamda')
    plt.ylabel('RMS Error')
    plt.show()