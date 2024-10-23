from ast import mod
import numpy as np
from matplotlib import pyplot as plt


class Regression:
    '''
        Classe para realizar regressão
    '''

    def __init__(self, file: str, perc_train: float):
        # Informações da tabela
        self.data = np.loadtxt(file)

        # Tamanho da base de dados e quantidade de variáveis
        self.N, self.p = self.data.shape

        # Variável independente
        self.x = self.data[:, 0]

        # Variável dependente
        self.y = self.data[:, 1]

        # Atualizando o tamanho para evitar erro
        self.x.shape = (self.N, 1)
        self.y.shape = (self.N, 1)

        self.perc_train = perc_train

    def los_traditional(self, x, y):
        # Retornar as estimativas dos coeficientes da função linear para o
        # modelo dos mínimos quadrados ordinários tradicional
        X = np.concatenate(
            (
                np.ones(x.shape),
                x
            ), axis=1
        )

        return np.linalg.pinv(X.T@X)@X.T@y

    def los_regular(self, x, y, regul):
        # Retornar as estimativas dos coeficientes da função linear para o
        # modelo dos mínimos quadrados ordinários regular
        X = np.concatenate(
            (
                np.ones(x.shape),
                x
            ), axis=1
        )
        return np.linalg.pinv(
            X.T@X + regul * np.eye(self.p))@X.T@y

    def avg_of_obs_values(self, y):
        # Retornar as estimativas dos coeficientes da função linear para o
        # modelo dos média de observações dos valores
        return np.mean(y, axis=0)

    def training(self, model, x_train, y_train, hiper=0):
        if model == self.los_regular.__name__:
            return self.los_regular(x_train, y_train, hiper)

        if model == self.los_traditional.__name__:
            return self.los_traditional(x_train, y_train)

        if model == self.avg_of_obs_values.__name__:
            return self.avg_of_obs_values(y_train)

    def montecarlo(self, model, hiper=0, R=500):
        results_of_model = []
        copy_data = np.copy(self.data)

        for _ in range(R):
            np.random.shuffle(copy_data)
            shuffled_x = copy_data[:, 0]
            shuffled_x.shape = (self.N, 1)
            shuffled_y = copy_data[:, 1]
            shuffled_y.shape = (self.N, 1)

            # Porcentagem de dados que serão utilizados para treinamento
            x_train = shuffled_x[:int(self.N*self.perc_train)]
            y_train = shuffled_y[:int(self.N*self.perc_train)]

            # Porcentagem de dados que serão utilizados para teste
            x_test = shuffled_x[int(self.N*self.perc_train):]
            y_test = shuffled_y[int(self.N*self.perc_train):]

            b_hat = self.training(model, x_train, y_train, hiper)

            X_test = np.concatenate(
                (
                    np.ones(x_test.shape),
                    x_test
                ), axis=1
            )
            y_hat = X_test@b_hat
            results_of_model.append(np.sum((y_hat - y_test)**2))

        return {"mean": np.mean(results_of_model),
                "max": np.max(results_of_model),
                "min": np.mean(results_of_model)}

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
    regul = [0.25, 0.5, 0.75, 1]
    print("LOS Traditional")
    print(regression.montecarlo('los_traditional'))
    print("LOS Regular")
    for r in regul:
        print(regression.montecarlo('los_regular', r))
    print("AVG Of Obs Values")
    print(regression.montecarlo('avg_of_obs_values'))
