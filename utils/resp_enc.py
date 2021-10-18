import pandas as pd
import numpy as np

class response_encoding:
  """
  This function is used to fit and transform the dataframe in one go.
  This is only made for binary classification problems.
  """
  def __init__(self,cols = [],target = 'Churn',alpha = 0, target_value = 1):
    """
    Parameters:
    -----------
    cols: list of categorical columns
    target: the target column
    alpha: the smoothing parameter
    target_value: the target value
    """
    self.cols = cols
    self.master_dict = {} #storing the original values
    self.alpha = alpha #smoothing parameter
    self.target = target
    self.target_value = 1
    
  def fit(self,df):
    alpha = self.alpha
    target = self.target
    for column in self.cols:
      unique_values = df[column].unique() #all unique values in that categorical column
      dict_values = {} #storing the response encoding values for target=1
      for value in unique_values:
        total = len(df[df[column]==value]) #the total no. of datapoints with 'value' catgeory
        sum_promoted = len(df[(df[column]==value) & (df[target]==self.target_value)]) #no. of all datapoints with category being 'value' and target=='yes'
        dict_values[value] = np.round((sum_promoted+alpha)/(total+alpha*len(unique_values)),2) #storing the obtained result in a dictionary
      dict_values['UNK']=0.5 #unknown categories that are not seen in train will be assigned a score of 0.5
      self.master_dict[column] = dict_values.copy() #storing the original values in a dictionary
    
    return None
    
  def transform(self,df):
    for column in self.cols:
      df[column] = df[column].map(self.master_dict[column]) #map the values in the column to the dictionary
    return df