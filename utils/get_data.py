import pandas as pd
import os
from sklearn.model_selection import train_test_split


def get_data(file_name:pd.DataFrame = '../data/WA_Fn-UseC_-Telco-Customer-Churn.csv', save = False)->pd.DataFrame:
    """Reads in the data from the specified file and returns a pandas dataframe.

    Args:
        file_name (pd.DataFrame, optional): Data csv file. Defaults to '../data/WA_Fn-UseC_-Telco-Customer-Churn.csv'.
        save (bool, optional): Whether to save the generated train and test file to csv. Defaults to False.

    Returns:
        pd.DataFrame: [description]
    """    
    files = ['train.csv', 'test.csv']
    if files not in os.listdir("../data"):
        df = pd.read_csv(file_name)

        #drop custoemr id because its not useful
        df = df.drop(['customerID'], axis=1)
        #changing the Churn value to binary: 1:churn, 0:no churn
        df['Churn'] = df['Churn'].map({'Yes':1, 'No':0})

        #TotalCharges is not needed
        df = df.drop(['TotalCharges'], axis=1)        

        #Senior citizen column should be object type
        df['SeniorCitizen'] = df['SeniorCitizen'].astype(str)

        train, test = train_test_split(
            df,
            test_size=0.2,
            stratify=df['Churn'],
            random_state=420)
        if save:
            train.to_csv('../data/train.csv')
            test.to_csv('../data/test.csv')
    else:
        train = pd.read_csv('../data/train.csv')
        test = pd.read_csv('../data/test.csv')
    
    return train, test

if __name__=='__main__':
    train, test = get_data()
    print(train.head())


