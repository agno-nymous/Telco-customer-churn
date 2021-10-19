import pandas as pd
import numpy as np
import get_data as g


class preprocess():
    def __init__(self) -> None:        
        self.col_to_remove = []
        self.get_obj_vars = []
        self.get_num_vars = []

    def read_data(self, file_name:str)->pd.DataFrame:
        """
        Method to read data from the csv file
        and return train and test data (80:20 split)
        Args:
            file_name (str): name of the csv file

        Returns:
            pd.DataFrame: train data and test data
        """        
        train, test = g.get_data(file_name=file_name, save=True)
        return train, test

    def fit(self, data:pd.DataFrame)->pd.DataFrame:
        """
        Method to fit the data 
        Args:
            data (pd.DataFrame): data to fit
        Returns:
            pd.DataFrame: fitted data
        """
        data = 
        return data