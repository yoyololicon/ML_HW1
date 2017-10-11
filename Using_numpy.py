import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == '__main__':
    data = pd.read_csv('data/4_test.csv')

    t = np.array(data['t'].values)
    x = np.array(data['x'].values)

    w = np.polyfit(x, t, 2)
    p = np.poly1d(w)
    y = p(x)
    rms = np.sqrt(np.sum((t-y)**2)/len(t))

    print rms

    xp = np.linspace(0, 10, 100)
    plt.plot(x, t, '.', xp, p(xp), '-')
    plt.ylim(-5, 10)
    plt.show()
