import numpy as np

class AdalineSGD(object):
    """Klasyfikator — ADAptacyjny LIniowy NEuron.
    Parametry
    ------------
    eta : zmiennoprzecinkowy
    Współczynnik uczenia (w zakresie pomiędzy 0.0 i 1.0). 
    n_iter : liczba całkowita
    Liczba przebiegów po zestawie uczącym.
    shuffle : wartość boolowska (domyślnie: True)
    Jeeli jest ustalona wartość True,
    tasuje dane uczące przed ka􏰁d􏰂 epok􏰂 w celu zapobiegnięcia cykliczności. 
    random_state : liczba całkowita
    Ziarno generatora liczb losowych słuące do inicjowania losowych wag.

    Atrybuty
    -----------
    w_ : jednowymiarowa tablica
    Wagi po dopasowaniu. 
    cost_ : lista
    Suma kwadratów błędów (wartość funkcji kosztu) w każdej epoce.
    """
    def __init__(self, eta=0.01, n_iter=10, shuffle=True, random_state=None):
        self.eta = eta
        self.n_iter = n_iter
        self.w_initialized = False
        self.shuffle = shuffle
        self.random_state = random_state
        self.rgen = np.random.RandomState(self.random_state)

    def fit(self, X, y):
        """ Trenowanie za pomocą danych uczących.
        Parametry
        ----------
        X : {tablicopodobny}, wymiary = [n_przykładów, n_cech]
        Wektory uczenia,
        gdzie n_przykładów oznacza liczbę przykładów, a n_cech — liczbę cech.
        y : tablicopodobny, wymiary = [n_przykładów] Wartości docelowe.
        Zwraca -------
        self : obiekt
        """
        self._initialize_weights(X.shape[1])
        self.cost_ = []
        for i in range(self.n_iter):
            if self.shuffle:
                X, y = self._shuffle(X, y)
            cost = []
            for xi, target in zip(X, y):
                cost.append(self._update_weights(xi, target))
            avg_cost = sum(cost) / len(y)
            self.cost_.append(avg_cost)
        return self

    def partial_fit(self, X, y):
        """Dopasowuje dane ucz􏰂ce bez ponownej inicjacji wag"""
        if not self.w_initialized:
            self._initialize_weights(X.shape[1])
        if y.ravel().shape[0] > 1:
            for xi, target in zip(X, y):
                self._update_weights(xi, target)
        else:
            self._update_weights(X, y)
        return self

    def _shuffle(self, X, y):
        """Tasuje dane uczące"""
        r = self.rgen.permutation(len(y))
        return X[r], y[r]

    def _initialize_weights(self, m):
        """Inicjuje wagi, przydzielając im małe, losowe wartości"""
        self.w_ = self.rgen.normal(loc=0.0, scale=0.01, size=1 + m)
        self.w_initialized = True

    def _update_weights(self, xi, target):
        """Wykorzystuje regułę uczenia Adaline do aktualizacji wag"""
        output = self.activation(self.net_input(xi))
        error = (target - output)
        self.w_[1:] += self.eta * xi.dot(error)
        self.w_[0] += self.eta * error
        cost = 0.5 * error**2
        return cost

    def net_input(self, X):
        """Oblicza całkowite pobudzenie"""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, X):
        """Oblicza liniową funkcję aktywacji"""
        return X

    def predict(self, X):
        """Zwraca etykietę klas po wykonaniu skoku jednostkowego"""
        return np.where(self.activation(self.net_input(X)) >= 0.0, 1, -1)
