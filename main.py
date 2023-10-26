from Perceptron import Perceptron
from AdalineGD import AdalineGD
from helpers import plot_decision_regions
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

s = os.path.join('https://archive.ics.uci.edu', 'ml',
    'machine-learning-databases',
    'iris','iris.data')

print('Adres URL:', s)

df = pd.read_csv(s, header=None, encoding='utf-8')
#print(df.tail())

y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', -1, 1)

X = df.iloc[0:100, [0,2]].values

# plt.scatter(X[:50, 0], X[:50, 1], 
#             color='red', marker='o', label='Setosa')
# plt.scatter(X[50:100, 0], X[50:100, 1],
#             color='blue', marker='x', label='Versicolor')
# plt.xlabel('Długość działki [cm]')
# plt.ylabel('Długość płatka [cm]')
# plt.legend(loc='upper left')
# plt.show()

# ppn = Perceptron(eta=0.1, n_iter=10)
# ppn.fit(X, y)

# plt.plot(range(1, len(ppn.errors_) +1), ppn.errors_,
#          marker='o')
# plt.xlabel('Epoki')
# plt.ylabel('Liczba aktualizacji')
# plt.show()

# plot_decision_regions(X, y, classifier=ppn)
# plt.xlabel('Długość działki [cm]')
# plt.ylabel('Długość płatka [cm]')
# plt.legend(loc='upper left')
# plt.show()

# fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(1000, 4))
# ada1 = AdalineGD(n_iter=10, eta=0.01).fit(X, y)
# ax[0].plot(range(1, len(ada1.cost_) + 1), 
#            np.log10(ada1.cost_), marker='o')
# ax[0].set_xlabel('Epoki')
# ax[0].set_ylabel('Log (suma kwadratów błędów)')
# ax[0].set_title('Adaline - Współczynnik uczenia 0.01')
# ada2 = AdalineGD(n_iter=1000, eta=0.0002).fit(X, y)
# ax[1].plot(range(1, len(ada2.cost_) + 1), 
#            np.log10(ada2.cost_), marker='o')
# ax[1].set_xlabel('Epoki')
# ax[1].set_ylabel('Log (suma kwadratów błędów)')
# ax[1].set_title('Adaline - Współczynnik uczenia 0.0002')
# plt.show()

X_std = np.copy(X)
X_std[:,0] = (X[:, 0] - X[:, 0].mean()) / X[:, 0].std()
X_std[:,1] = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std()

ada_gd = AdalineGD(n_iter=30, eta=0.01).fit(X_std, y)

plot_decision_regions(X_std, y, classifier=ada_gd)
plt.title('Adaline - Gradient prosty')
plt.xlabel('Długość działki [standaryzowana]')
plt.ylabel('Długość płatka [standaryzowana]')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()

plt.plot(range(1, len(ada_gd.cost_) + 1), ada_gd.cost_, marker='o') 
plt.xlabel('Epoki')
plt.ylabel('Suma kwadratów błędów')
plt.tight_layout()
plt.show()
