import numpy as np
from matplotlib import pyplot as plt


class Regression:
    '''
        Classe para realizar regressão
    '''

    def __init__(
            self, file_name: str, percent_train: float,
    ):
        # Informações da tabela
        self.data = np.loadtxt(file_name)

        # Tamanho da base de dados e quantidade de variáveis
        self.N, self.p = self.data.shape

        # Variável independente
        self.x = self.data[:, 0]

        # Variável dependente
        self.y = self.data[:, 1]

        # Atualizando o tamanho para evitar erro
        self.x.shape = (self.N, 1)
        self.y.shape = (self.N, 1)

        self.percent_train = percent_train

    def los_traditional(self, x, y):

        # Definindo matriz X, utilizando o x para teino do modelo considerando
        # o B0
        X = np.concatenate(
            (
                np.ones(x.shape),
                x
            ), axis=1
        )

        # Retornar as estimativas dos coeficientes da função linear para o
        # modelo dos mínimos quadrados ordinários tradicional
        w = np.linalg.pinv(X.T@X)@X.T@y
        return w

    def los_regular(self, x, y, regul):
        # Retornar as estimativas dos coeficientes da função linear para o
        # modelo dos mínimos quadrados ordinários regular
        X = np.concatenate(
            (
                np.ones(x.shape),
                x
            ), axis=1
        )
        w = np.linalg.pinv(
            X.T@X + regul * np.eye(self.p))@X.T@y
        return w

    def avg_of_obs_values(self):
        return np.mean(self.y_train, axis=0)

    def montecarlo(self, model, hiper=0, R=500):
        results_of_model = []
        for _ in range(R):
            # Porcentagem de dados que serão utilizados para treinamento
            x_train = self.x[:int(self.N*self.percent_train)]
            y_train = self.y[:int(self.N*self.percent_train)]

            # Porcentagem de dados que serão utilizados para teste
            x_test = self.x[int(self.N*self.percent_train):]
            y_test = self.y[int(self.N*self.percent_train):]

            result_model = model(x_train, y_train, hiper) \
                if model == self.los_regular else model(x_train, y_train)

            print(result_model)
            results_of_model.append(result_model)

    def show_graphic(self):
        # Mostrar gráfico
        plt.figure(0)
        plt.scatter(self.x, self.y, edgecolors='k')
        plt.xlabel('Velocidade do vento')
        plt.ylabel('Potência gerada pelo aerogerador')
        plt.show()


if __name__ == '__main__':
    regression = Regression('aerogerador.dat', 0.8)
    regression.show_graphic()
    regul = [0, 0.25, 0.5, 0.75, 1]
