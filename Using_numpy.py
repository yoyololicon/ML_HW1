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
        w = np.polyfit(train_x, train_t, order)
        p = np.poly1d(w)

        train_y = p(train_x)
        test_y = p(test_x)

        train_elist.append(np.sqrt(np.mean((train_y - train_t) ** 2)))
        test_elist.append(np.sqrt(np.mean((test_y - test_t) ** 2)))

    plt.plot(train_elist, marker='o')
    plt.plot(test_elist, 'r-', marker='o')
    plt.xlim(-1, 10)
    plt.xlabel('M')
    plt.ylabel('RMS Error')
    plt.show()
