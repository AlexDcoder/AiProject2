import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


class Classification:
    def __init__(self, file: str,  perc_train: float):
        # Informações da tabela
        self.data = np.loadtxt(file, delimiter=',')

        # Tamanho da base de dados e quantidade de variáveis
        self.N, self.p = self.data.shape

        # Categorias
        self.cat = {
            'Neutro': 0,
            'Sorriso': 1,
            'Rabugento': 2,
            'Sobrancelhas levantadas': 3,
            'Surpreso': 4
        }

        # Cores correspondentes das categorias
        self.colors = {
            'Neutro': 'green',
            'Sorriso': 'yellow',
            'Rabugento': 'red',
            'Sobrancelhas levantadas': 'pink',
            'Surpreso': 'purple'
        }

        # Variáveis independentes
        self.X1 = self.data[0, :]
        self.X2 = self.data[1, :]

        # Variável dependente
        self.Y = self.data[2, :]

        # Porcentagem de treino
        self.perc_train = perc_train

    def training(self, model, hiper=0):
        pass

    def montecarlo(self, model, hiper=0, R=500):
        results_model = []
        copy_data = np.copy(self.data)

        for _ in range(R):
            # Embaralhando os dados e divide os dados entre treino e testes
            np.random.shuffle(copy_data)

            # Porcentagem de dados que serão utilizados para treinamento
            # Porcentagem de dados que serão utilizados para teste

        return [np.mean(results_model),
                np.std(results_model),
                np.max(results_model),
                np.mean(results_model)]

    def show_graphic(self):
        # Mostrar gráfico
        plt.figure(1)
        plt.title('Classificação de Emoções')
        for name in self.cat.keys():
            plt.scatter(
                self.X1.T[self.Y.T == self.cat[name]],
                self.X2.T[self.Y.T == self.cat[name]],
                color=self.colors[name], edgecolor='k',
                label=name,
            )

        plt.xlabel('Corrugador do Supercílio')
        plt.ylabel('Zigomático Maior')
        plt.show()


if __name__ == '__main__':
    classification = Classification('EMGsDataset.csv', 0.8)
    classification.show_graphic()
    print(classification.X1)
    print(classification.X2)
    print(classification.Y)
    # data_models = {
    #     'Modelos': [
    #         'MQO tradicional',
    #         'Classificador Gaussiano Tradicional',
    #         'Classificador Gaussiano (Cov iguais)',
    #         'Classificador Gaussiano (Cov Agregada)',
    #         'Classificador de Bayes Ingênuo',
    #         'Classificador Gaussiano Regularizado (0,25)',
    #         'Classificador Gaussiano Regularizado (0,5)',
    #         'Classificador Gaussiano Regularizado (0,75)',
    #     ],
    #     'Média': [],
    #     'Desvio-Padrão': [],
    #     'Maior Valor': [],
    #     'Menor Valor': [],
    # }
