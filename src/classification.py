import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


class Classification:
    def __init__(self, data):
        self.data = data
        b1 = ''

    def show_grpah(self):
        pass


if __name__ == '__main__':
    data = np.loadtxt('EMGsDataset.csv', delimiter=',')
    c = Classification(data)
