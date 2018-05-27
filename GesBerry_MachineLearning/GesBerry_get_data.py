# imports
import numpy as np # -------------------------------------- vers. numpy:  1.14.0
import pandas as pd # ------------------------------------- vers. pandas: 0.22.0

import sys, os

def import_recorded_datasets(folder_name):
    """(str) --> np.array, np.array
    
    Reads in the *.csv files from the dataset at a given path-location and returns their values as
    numpy arrays.
    
    args:
        * path (str): path to folder
    
    returns:
        * X_raw (np.array): raw unprocessed dataset - matrix
        * y (np.array)    : label vector    
    """
    
    path = os.getcwd() + '/data/' + folder_name
    
    X_raw = pd.read_csv(path + '/' + 'raw_data.csv', index_col=0).values
    y     = pd.read_csv(path + '/' + 'label.csv'   , header=None).values.ravel()
    
    return X_raw,y

