import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.odr import Model, ODR, RealData


class DataTable:
    def __init__(self, data_table, main_columns_indxs=[0, 1, 2, 3]):
        self.main_columns_indxs = main_columns_indxs
        self.data_table = data_table

        self.x, self.delta_x, self.y, self.delta_y = self.choose_columns(self.data_table, main_columns_indxs)

    @classmethod
    def from_excel(cls, file_path, main_columns_indxs=[0, 1, 2, 3]):
        data_table = cls.get_excel(file_path)

        return cls(data_table, main_columns_indxs)

    @classmethod
    def from_list(cls, data_list, column_names, main_columns_indxs=[0, 1, 2, 3]):
        data_table = pd.DataFrame(data_list, columns=column_names)
        return cls(data_table, main_columns_indxs)

    @staticmethod
    def get_excel(file_path, sheet_number=0):
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

    def __str__(self):
        return f'${self.data_table}'
