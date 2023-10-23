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
print(df.tail())

y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', -1, 1)

X = df.iloc[0:100, [0,2]].values

plt.scatter(X[:50, 0], X[:50, 1], 
            color='red', marker='o', label='Setosa')
plt.scatter(X[50:100, 0], X[50:100, 1],
            color='blue', marker='x', label='Versicolor')
plt.xlabel('Długość działki [cm]')
plt.ylabel('Długość płatka [cm]')
plt.legend(loc='upper left')
plt.show()

ppn = Perceptron(eta=0.1, n_iter=10)
ppn.fit(X, y)

plt.plot(range(1, len(ppn.errors_) +1), ppn.errors_,
         marker='o')
plt.xlabel('Epoki')
plt.ylabel('Liczba aktualizacji')
plt.show()

plot_decision_regions(X, y, classifier=ppn)
plt.xlabel('Długość działki [cm]')
plt.ylabel('Długość płatka [cm]')
plt.legend(loc='upper left')
plt.show()
