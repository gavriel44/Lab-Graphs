class DataTable:
    def __init__(self, data_sheet, columns=[0, 1, 2, 3]):
        self.data_sheet = data_sheet
        self.x, self.delta_x, self.y, self.delta_y = self.choose_columns(data_sheet, columns)

    def choose_columns(data, columns):
        x = data.iloc[:, columns[0]]
        delta_x = data.iloc[:, columns[1]]
        y = data.iloc[:, columns[2]]
        delta_y = data.iloc[:, columns[3]]
        return x, delta_x, y, delta_y

    def __repr__(self):
        return f"DataTable(x={self.x}, y={self.y}, delta_x={self.delta_x}, delta_y={self.delta_y})"
