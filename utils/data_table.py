import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.odr import Model, ODR, RealData


class DataTable:
    def __init__(self, file_path, main_columns_indxs=[0, 1, 2, 3]):
        self.main_columns_indxs = main_columns_indxs

        self.data_table = self.get_excel(file_path)
        self.x, self.delta_x, self.y, self.delta_y = self.choose_columns(self.data_table, main_columns_indxs)

    def get_excel(self, file_path, sheet_number=0):
        # file_path = r"/content/FreeFallGroupD.xls" # Replace with your file path: r"/contents/<your file name>.xlsx"
        return pd.read_excel(file_path, sheet_name=sheet_number)

    def choose_columns(self, data, columns):
        x = data.iloc[:, columns[0]]
        delta_x = data.iloc[:, columns[1]]
        y = data.iloc[:, columns[2]]
        delta_y = data.iloc[:, columns[3]]
        return x, delta_x, y, delta_y

    def get_x_label(self):
        return f'{self.data_table.columns[self.main_columns_indxs[0]]}'

    def get_y_label(self):
        return f'{self.data_table.columns[self.main_columns_indxs[2]]}'

    def __repr__(self):
        return f"DataTable(x={self.x}, y={self.y}, delta_x={self.delta_x}, delta_y={self.delta_y})"

    def columns(self):
        return self.data_table.columns
