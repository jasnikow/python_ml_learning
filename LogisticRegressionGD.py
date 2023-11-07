import numpy as np

class LogisticRegressionGD(object):
    """Klasyfikator regresji logistycznej wykorzystuj􏰏cy metod􏰐 gradientu prostego
    Parametry
    ------------
    eta : zmiennoprzecinkowy
        Współczynnik uczenia (pomi􏰐dzy 0,0 a 1,0) n_iter : liczba całkowita
        Przebiegi po zestawie danych ucz􏰏cych. random_state : liczba całkowita
        Ziarno generatora liczb losowych do losowej inicjacji wag.
    Atrybuty
    -----------
    w_ : tablica jednowymiarowa
        Wagi po dopasowaniu. 
    cost_ : lista
        Warto􏰔􏰕 funkcji kosztu (logistyczna) w ka􏰖dej epoce.
    """
    def __init__(self, eta=0.05, n_iter=100, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        """ Dopasowanie danych ucz􏰏cych.
        Parametry
        ----------
        X : {tablicopodobny}, wymiary = [n_przykładów, n_cech]
            Wektory uczenia, gdzie n_przykładów oznacza liczb􏰐 przykładów, a
            n_cech — liczb􏰐 cech.
        y : tablicopodobny, wymiary = [n_przykładów]
            Warto􏰔ci docelowe.
        Zwraca 
        -------
        self : obiekt
        """
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size = 1 + X.shape[1])

        self.cost_ = []

        for i in range(self.n_iter):
            net_input = self.net_input(X)
            output = self.activation(net_input)
            errors = (y - output)
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()

            # Zwró􏰕 uwag􏰐, 􏰖e obliczamy teraz ‘koszt’ logistyczny, a nie # sum􏰐 kwadratów bł􏰐dów.
            cost = (-y.dot(np.log(output)) - ((1 - y).dot(np.log(1 - output))))
            self.cost_.append(cost)
            return self
        
        def net_input(self, X):
            """Obliczanie pobudzenia całkowitego"""
            return np.dot(X, self.w_[1:]) + self.w_[0]
        
        def activation(self, z):
            """Obliczanie logistycznej, sigmoidalnej funkcji aktywacji"""
            return 1. / (1. + np.exp(-np.clip(z, -250, 250)))
        
        def predict(self, X):
            """Zwraca etykiet􏰐 klasy po skoku jednostkowym"""
            return np.where(self.net_input(X) >= 0.0, 1, 0)
            # equivalent to:
            # return np.where(self.activation(self.net_input(X))
            #                            >= 0.5, 1, 0)