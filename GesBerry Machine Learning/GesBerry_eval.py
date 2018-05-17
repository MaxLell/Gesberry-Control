# imports
import numpy as np # -------------------------------------- vers. numpy:  1.14.0
import pandas as pd # ------------------------------------- vers. pandas: 0.22.0
import sys, os

import matplotlib.pyplot as plt # -------------------------- vers. matplotlib: 2.1.2
import seaborn as sn # ------------------------------------- vers. seaborn: 0.8.1
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split # ---- vers. scikit-learn: 0.19.1

from GesBerry_get_data  import import_recorded_datasets
from GesBerry_prep      import preprocess_raw_data

from sklearn.ensemble import RandomForestClassifier

def conf_matrix(clf, X_test, y_test):
    '''
    (clf, np.array, np.array) --> None
    
    plots a heatmap of the confusion matrix of the current train-test-split.
    
    args:
        * clf : trained classifier
        * X_test (np.array) : preprocessed test-dataset
        * y_test (np.array) : test-label vector
    
    returns:
        * None
    
    '''
    label = np.unique([y_test])
    
    # make prediction
    y_pred = clf.predict(X_test)
    
    # confusion matrix
    cm = confusion_matrix(y_test, y_pred, label)
    
    plt.figure(figsize = (7,5.5))
    pal = sn.diverging_palette(220, 20, n=7, as_cmap = True)
    
    df_cm = pd.DataFrame(cm, index = [i for i in label], columns = [i for i in label])
    ax = sn.heatmap(df_cm, annot=True, cmap = pal, square=True, fmt = 'g')
    ax.set(xlabel='Predicted values', ylabel='True values')
    

def clf_acc(clf, X_test, y_test):
    '''(clf, np.array, np.array) --> None
    
    Prints the accuracy score for the current train-test-split
    
    args:
        * clf : trained classifier
        * X_test (np.array) : preprocessed test-dataset
        * y_test (np.array) : test-label vector
    
    returns:
        * None
    '''
    
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print ('Accuracy of the current Train_test-split --------- : ' , acc)

def cross_val(clf, X, y):
    """(clf, np.array, np.array) -> None
    
    calculates and prints a 10 fold crossval score.
    
    args:
        * clf: trained classifer
    
    return:
        * None
    """
    
    scores = cross_val_score(clf, X, y, cv = 10)
    print(clf.__class__.__name__, '--->',"Crossval score Accuracy:  %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    
def x_y_test(dataset1, dataset2, frame_size = 140):
    """(str, str) -> None
    
    performs an (X1-X2)-Test and plots evaluation scores: The Machine Learning algorithm is trained on dataset1
    and tested on dataset2. This function prints an accuracy score and a confusion matrix.
    
    args:
        * dataset1 (str) : folder name of dataset 1
        * dataset2 (str) : folder name of dataset 2
    
    returns:
        * None
    """

    # 1. Get Data
    X_raw_train, y_train  = import_recorded_datasets(dataset1) # ------ load training dataset and training labels
    X_raw_test,  y_test   = import_recorded_datasets(dataset2) # ------ load test dataset and test labels

    # 2. Preprocess Data
    X_train = preprocess_raw_data(X_raw_train, frame_size) # ---------- preprocess raw dataset
    X_test  = preprocess_raw_data(X_raw_test , frame_size) # ---------- preprocess raw dataset

    # 3 Machine Learning
    clf = RandomForestClassifier(bootstrap = True, max_depth = 30, max_features = 4, n_estimators = 250)
    clf.fit(X_train, y_train)

    # 4.Evaluation
    conf_matrix(clf, X_test, y_test) # --------------- plot Confusion Matrix
    clf_acc(clf, X_test, y_test) # ------------------- print Accuracy Score (single dataset split)
    
def x_test(dataset, frame_size = 140):
    """(str) -> None
    
    performs a (X)-Test and plots evaluations scores: The Machine Learning algorithm is trained and validated on
    one dataset. This function: 
    
    1. catches the data from the given folder and performs a single train-test split on the dataset
    2. preprocesses the data (X_train, X_test, X)
    3. trains a Machine Learning classifier
    4. prints a 10-fold-cross validation score, 
       prints an accuracy score of the current split, 
       plots a confusion matrix of the current split.
       
    args:
        * dataset (str)     : name of the dataset's folder
        * [frame_size (int) : length of the recording window]
        
    returns:
        * None
    """
    
    ################### 1. Get Data #################
    X_raw,y = import_recorded_datasets(dataset) # ---------- Function loads the data and the labels
    X_train_raw, X_val_raw, y_train, y_val = train_test_split(X_raw, y, test_size=0.3) # train-val-split

    ################## 2. Preprocess Data ###########
    X_train = preprocess_raw_data(X_train_raw, frame_size)
    X_val   = preprocess_raw_data(X_val_raw , frame_size)
    X       = preprocess_raw_data(X_raw, frame_size) # ----- preprocess entire dataset for crossval score

    ################# 3. Machine Learning ###########
    clf = RandomForestClassifier(bootstrap = True, max_depth = 30, max_features = 4, n_estimators = 250)


    ################# 4.Evaluation ##################
    cross_val(clf, X, y) # ------------------------- 10-fold-cross-valiadation score

    clf.fit(X_train, y_train) # -------------------- train classifier on train-val-dataset split
    conf_matrix(clf, X_val, y_val) # --------------- plot Confusion Matrix
    clf_acc(clf, X_val, y_val) # ------------------- print Accuracy Score (single dataset split)
