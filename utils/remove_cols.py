import pandas as pd
import numpy as np
import os
import utils.get_data as get_data
from scipy.stats import chi2_contingency

def get_obj_cols(data: pd.DataFrame) -> list:
    """
    Returns a list of columns that are of object dtype.

    Args:
        df (pd.DataFrame): train dataframe

    Returns:
        list: list of columns
    """    

    return [i for i in data.columns if data[i].dtype == 'object']

def get_int_cols(data: pd.DataFrame) -> list:
    """
    Returns a list of columns that are of int dtype.

    Args:
        df (pd.DataFrame): train dataframe

    Returns:
        list: list of columns
    """    

    return [i for i in data.columns if data[i].dtype != 'object']

def get_significant_categorical(
    data: pd.DataFrame,
    columns:list,
    target:str = 'Churn',
    threshold:float = 0.05,
    )-> list:
    """This function returns all the categorical columns which have significant difference with target column unique values using
    Chi-squared test and also those columns which are not significant

    Args:
        data (pd.DataFrame): trian dataframe
        columns (list): columns which contain categroical data
        target (str, optional): Target column. Defaults to 'Churn'.
        threshold (float, optional): Threshold to categorize non-significant. Defaults to 0.05.

    Returns:
        [list]: list of significant columns and non-significant columns
    """    

    significant = np.array([])
    non_significant = np.array([])
    for col in columns:
        chi2, p, dof, expected = chi2_contingency(pd.crosstab(data[col], data[target]))
        if p < threshold:
            significant = np.append(significant, col)
        else:
            non_significant = np.append(non_significant, col)

    return significant, non_significant

def main(
    target: str = 'Churn',
    ) -> pd.DataFrame:
    """This function removes the unimportant columns

    Args:
        target (str, optional): Target column. Defaults to 'Churn'.

    Returns:
        pd.DataFrame: train and test dataframes with removed columns
    """
    train, test = get_data.get_data()
    obj_cols = get_obj_cols(train)
    int_cols = get_int_cols(train)
    significant_obj, non_significant_obj = get_significant_categorical(train, obj_cols, target)

    for col in non_significant_obj:
        train.drop(col, axis=1, inplace=True)
        test.drop(col, axis=1, inplace=True)

    return train, test