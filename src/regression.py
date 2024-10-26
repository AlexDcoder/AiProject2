import numpy as np
import pandas as pd
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

        # Porcentagem de treino
        self.perc_train = perc_train

    @staticmethod
    def los_traditional(x, y):
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

    @staticmethod
    def avg_of_obs_values(y):
        # Retornar as estimativas dos coeficientes da função linear para o
        # modelo dos média de observações dos valores
        return np.mean(y, axis=0)

    def training(self, model, x_train, y_train, hiper=0):
        # Treinamento com base no modelo selecionado
        if model == self.los_regular.__name__:
            return self.los_regular(x_train, y_train, hiper)

        if model == self.los_traditional.__name__:
            return self.los_traditional(x_train, y_train)

        if model == self.avg_of_obs_values.__name__:
            return self.avg_of_obs_values(y_train)

    def montecarlo(self, model, hiper=0, R=500):
        results_model = []
        copy_data = np.copy(self.data)

        for _ in range(R):
            # Embaralhando os dados e divide os dados entre treino e testes
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

            if model != self.avg_of_obs_values.__name__:
                # Valores estimados dos coeficientes angulares
                b_hat = self.training(model, x_train, y_train, hiper)

                # Matriz dos valores de teste
                X_test = np.concatenate(
                    (
                        np.ones(x_test.shape),
                        x_test
                    ), axis=1
                )

                # Valores estimados com o teste
                y_hat = X_test@b_hat

                # Adicionando os resultados do modelo da rodada atual
                results_model.append(np.sum((y_hat - y_test)**2))
            else:
                y_hat = self.training(model, x_train, y_train, hiper)
                # Adicionando os resultados do modelo da rodada atual
                results_model.append(np.sum((y_hat - y_test)**2))

        return [np.mean(results_model),
                np.std(results_model),
                np.max(results_model),
                np.mean(results_model)]

    def show_graphic(self):
        # Mostrar gráfico
        plt.figure(1)
        plt.scatter(self.x, self.y, edgecolors='k', alpha=0.5)
        plt.xlabel('Velocidade do vento')
        plt.ylabel('Potência gerada pelo aerogerador')
        plt.show()


if __name__ == '__main__':
    regression = Regression('aerogerador.dat', 0.8)
    regression.show_graphic()

    data_models = {
        'Modelos': ['Média da variável dependente',
                    'MQO tradicional',
                    'MQO regularizado (0,25)',
                    'MQO regularizado (0,5)',
                    'MQO regularizado (0,75)',
                    'MQO regularizado (1)',
                    ],
        'Média': [],
        'Desvio-Padrão': [],
        'Maior Valor': [],
        'Menor Valor': [],
    }

    # AVG Of Obs Values
    avg_obs = regression.montecarlo('avg_of_obs_values')
    data_models['Média'].append(avg_obs[0])
    data_models['Desvio-Padrão'].append(avg_obs[1])
    data_models['Maior Valor'].append(avg_obs[2])
    data_models['Menor Valor'].append(avg_obs[3])

    # LOS Traditional
    los_t = regression.montecarlo('los_traditional')
    data_models['Média'].append(los_t[0])
    data_models['Desvio-Padrão'].append(los_t[1])
    data_models['Maior Valor'].append(los_t[2])
    data_models['Menor Valor'].append(los_t[3])
    # LOS Regular
    regul = [0.25, 0.5, 0.75, 1]
    for r in regul:
        los_r = regression.montecarlo('los_regular', r)
        data_models['Média'].append(los_r[0])
        data_models['Desvio-Padrão'].append(los_r[1])
        data_models['Maior Valor'].append(los_r[2])
        data_models['Menor Valor'].append(los_r[3])

    df_models = pd.DataFrame(data_models)
    print(df_models)
