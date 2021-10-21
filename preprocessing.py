import pandas as pd
import numpy as np

# import matplotlib.pyplot as plt
# import seaborn as sns

# import all models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import sklearn.model_selection as model_selection
import xgboost as xgb
import lightgbm as lgb

# import all metrics
import sklearn.metrics as metrics

import warnings

warnings.filterwarnings("ignore")


# create a class response_encoding which fit and transform the categorical columns


class response_encoding:
    """
    This function is used to fit and transform the dataframe in one go.
    This is only made for binary classification problems.
    """

    def __init__(self, cols, target="Churn", alpha=0, target_value=1):
        """
        Parameters:
        -----------
        cols: list of categorical columns
        target: the target column
        alpha: the smoothing parameter
        target_value: the target value
        """
        self.cols = cols
        self.master_dict = {}  # storing the original values
        self.alpha = alpha  # smoothing parameter
        self.target = target
        self.target_value = 1

    def fit(self, df):
        alpha = self.alpha
        target = self.target
        for column in self.cols:
            unique_values = df[
                column
            ].unique()  # all unique values in that categorical column
            dict_values = {}  # storing the response encoding values for target=1
            for value in unique_values:
                total = len(
                    df[df[column] == value]
                )  # the total no. of datapoints with 'value' catgeory
                sum_promoted = len(
                    df[(df[column] == value) & (df[target] == self.target_value)]
                )  # no. of all datapoints with category being 'value' and target=='yes'
                dict_values[value] = np.round(
                    (sum_promoted + alpha) / (total + alpha * len(unique_values)), 2
                )  # storing the obtained result in a dictionary
            dict_values[
                "UNK"
            ] = 0.5  # unknown categories that are not seen in train will be assigned a score of 0.5
            self.master_dict[
                column
            ] = dict_values.copy()  # storing the original values in a dictionary

        return None

    def transform(self, df):
        for column in self.cols:
            df[column] = df[column].map(
                self.master_dict[column]
            )  # map the values in the column to the dictionary
        return df


def get_data():
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    return train, test


def get_cols(data):
    get_obj_cols = [
        col for col in data.columns if data[col].dtype == "object" & col != "Churn"
    ]
    get_int_cols = [
        col for col in data.columns if data[col].dtype != "object" & col != "Churn"
    ]
    return get_obj_cols, get_int_cols


def get_data_with_encoding(train, test):
    # get data
    train, test = get_data()
    # get columns
    get_obj_cols, get_int_cols = get_cols(train)

    # define the response encoding class
    response_encoding_obj = response_encoding(
        get_obj_cols, target="Churn", alpha=0.1, target_value=1
    )

    # fit the response encoding class
    response_encoding_obj.fit(train)

    # transform the data
    train = response_encoding_obj.transform(train)
    test = response_encoding_obj.transform(test)

    return train, test, get_obj_cols, get_int_cols


def get_models(weight):
    # define the models
    models = {
        "Decision Tree": DecisionTreeClassifier(class_weight="balanced"),
        "Random Forest": RandomForestClassifier(
            class_weight="balanced_subsample", n_jobs=-1, n_estimators=50, max_depth=10
        ),
        "XGBoost": xgb.XGBClassifier(
            scale_pos_weight=weight, use_label_encoder=False, n_jobs=-1
        ),
        "LightGBM": lgb.LGBMClassifier(class_weight="balanced", n_jobs=-1),
    }
    return models


def modelling():
    # get data
    train, test = get_data()

    # get response encoded data
    train, test, get_obj_cols, get_int_cols = get_data_with_encoding(train, test)

    # define x and y
    x_train = train.drop(["Churn"], axis=1)
    y_train = train["Churn"]
    x_test = test.drop(["Churn"], axis=1)
    y_test = test["Churn"]
    del train, test

    # define the cv
    cv = model_selection.RepeatedStratifiedKFold(
        n_splits=3, n_repeats=2, random_state=42
    )

    # define the weight
    # for xgboost
    # weight = no. of negative classes/no. of positive classes
    weight = (y_train == 0).sum() / (y_train == 1).sum()

    # get the models
    models = get_models(weight)

    for model_name, model in models.items():
        scores = model_selection.cross_val_score(
            model, x_train, y_train, cv=cv, scoring="f1"
        )
        print(model_name)
        mean = np.mean(scores)
        # print(f"mean f1 score: {mean}")
        print(mean)
        print("\n")


if __name__ == "__main__":
    modelling()
