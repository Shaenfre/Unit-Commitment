import pandas as pd
import numpy as np

class file_reading:
    def __init__(self, file_name):
        self.excel_file = pd.ExcelFile(file_name)

    def sheet_names(self):
        self.sheet_names = self.excel_file.sheet_names
        return self.sheet_names

    def open_file(self, file_name, sheet_name):
        excel_sheet_df = pd.read_excel(file_name, sheet_name)
        return excel_sheet_df
    #
    # #convert a particular column to a numpy array
    # def to_numpy(self, col_name):
    #     return self.excel_sheet_df[col_name].to_numpy()
    #
    #convert all columns into numpy array and store them in a dictionary with column name as the key
    def to_numpy_all(self, sheet_df):
        datasheet_dict = {}
        for col in sheet_df.columns:
            datasheet_dict[col] = sheet_df[col].to_numpy()
        return datasheet_dict


#Reading bus data
excel_fileReading = file_reading('IEEE9.xlsx')


