import numpy as np
from matplotlib import pyplot as plt


class Regression:
    def __init__(self, file_name: str):
        self.data = np.loadtxt(file_name)
        self.N = self.data.shape[0]
        self.x = self.data[:, 0]
        self.x.shape = (self.N, 1)
        self.y = self.data[:, 1]
        self.y.shape = (self.N, 1)
        self.w = self.define_weights()

    def define_weights(self):
        X = np.concatenate(
            (
                np.ones((self.N, 1)),
                self.x),
            axis=1
        )
        return np.linalg.pinv(X.T@X)@X.T@self.y

    def function_label(self):
        pass

    def show_graphic(self):
        fig, ax = plt.subplots()
        ax.scatter(self.x, self.y, edgecolors='k')
        plt.xlabel('Velocidade do vento')
        plt.ylabel('Potência gerada pelo aerogerador')
        plt.show()


if __name__ == '__main__':
    regression = Regression('aerogerador.dat')
    regression.show_graphic()
