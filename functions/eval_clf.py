####################### Copyright ##########################
##The gesture-control-interface (GCI) device library code is placed under the MIT license
##Copyright (c) 2017 Maximilian Lell (maximilian.lell@gmail.com)
##
##Permission is hereby granted, free of charge, to any person obtaining a copy
##of this software and associated documentation files (the "Software"), to deal
##in the Software without restriction, including without limitation the rights
##to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
##copies of the Software, and to permit persons to whom the Software is
##furnished to do so, subject to the following conditions:
##
##The above copyright notice and this permission notice shall be included in
##all copies or substantial portions of the Software.
##
##THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
##IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
##FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
##AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
##LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
##OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
##THE SOFTWARE.

####################### descriptions ########################
# The purpose of this script is to evaluate the classifiers performance
# to do so a heatmap of a confusion matrix is generated and the classifiers
# accuracy is generated.

####################### imports #############################
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt
import seaborn as sn

from preprocessing import preprocess_raw_dataset

import numpy as np
import pandas as pd

import timeit
####################### functions ###########################
def cf_matrix(clf, X_test, y_test, small_plot = False):
    '''
    (clf, np.array, np.array, str) --> None
    
    plots a heatmap of the confusion matrix of the current train-test-split.
    '''
    label = np.unique([y_test])
    
    # make prediction
    y_pred = clf.predict(X_test)
    
    # confusion matrix
    cm = confusion_matrix(y_test, y_pred, label)
    
    plt.figure(figsize = (7,5.5))
    pal = sn.diverging_palette(220, 20, n=7, as_cmap = True)
    
    if small_plot == False:
        df_cm = pd.DataFrame(cm, index = [i for i in label], columns = [i for i in label])
        ax = sn.heatmap(df_cm, annot=True, cmap = pal, square=True, fmt = 'g')
        ax.set(xlabel='Predicted values', ylabel='True values')
    else:
        df_cm = pd.DataFrame(cm)
        ax = sn.heatmap(df_cm, annot=True, cmap = pal, square=True, fmt = 'g', cbar = False)

def clf_acc(clf, X_test, y_test):
    '''(clf, np.array, np.array) --> None
    
    Prints the accuracy score for the current train-test-split
    '''
    
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print ('accuracy of the current Train_test-split' , acc)
    
def cross_val(clf, X, y):
    '''(clf, np.array, np.array) --> None
    
    calculates and prints a 10 fold crossval score.
    '''
    
    scores = cross_val_score(clf, X, y, cv = 10)
    print("Crossval score Accuracy:                 %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
          
def clf_speed(clf, X, y):
    '''(clf, np.array, np.array) --> None
    
    prints the the worst case scenario - 
    the maximal duration of the classification durance
    '''
    speed = np.array([])
    for i in range(len(y)):
        start_time = timeit.default_timer()
        y_pred = clf.predict(X[i,:].reshape(1, -1))
        end_time = timeit.default_timer()
        
        time = end_time - start_time
        speed = np.append(speed, time)
    #print ('\n')
    print('max clf time                            ', np.max (speed), '[sec]')

def preprocess_speed(X_raw, frame_size, prepmode):
    '''(np.array, int, str) --> None
    prints the worst case scenario -
    the maximal duration of the preprocessing time.
    '''
    speed = np.array([])
    for i in range(X_raw.shape[0]):
        start_time = timeit.default_timer()
        dump = preprocess_raw_dataset(X_raw[0,:].reshape(1,-1), frame_size, prepmode)
        end_time = timeit.default_timer()
        time = end_time - start_time
        speed = np.append(speed, time)
    print('max preprocess time                     ', np.max (speed), '[sec]')
    
def eval_(clf, X_test, y_test, small_plot = False, speed_test = False, 
                       X = None, y = None, X_raw = None, prepmode = None, frame_size = None):
    
    '''(clf, np.array, np.array, np.array, np.array) --> None
    
    Plots a confusion matrix of the current train-test split
    prints the accuracy of the current train-test split
    
    prints the 10 fold cross validation score
    prints the classification speed
    '''
    if frame_size == None:
        frame_size = 140
    
    if speed_test == True:
        # 1. classifier speed; preprocess speed
        clf_speed(clf, X, y)
        preprocess_speed(X_raw, frame_size, prepmode)
    
    # 2. confusion matrix
    cf_matrix(clf, X_test, y_test, small_plot)
    
    # 3. accurcy score current Train-Test split:
    clf_acc(clf, X_test, y_test)
    

    
def get_datasets(path):
    '''(str) --> np.array[130,980], np.array[130,1]
    returns the dataset (X) and the labels (y) from a given path out of a *.csv-file
    '''
    y = pd.read_csv(path + '/' + 'label.csv'   , header=None).values.ravel()
    X = pd.read_csv(path + '/' + 'raw_data.csv', index_col=0).values
    
    return y,X