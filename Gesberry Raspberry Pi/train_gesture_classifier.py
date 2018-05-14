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

################# imports ###################
import numpy as np
import pandas as pd

import os
import sys
sys.path.append(os.getcwd() + '/' + 'functions')

from record_data import record_labelled_dataset
from preprocessing import preprocess_raw_dataset
from train_clf import train_clf
from sklearn.externals import joblib


################# globals ####################
frame_size = 140 # window_size = 1.4 sec
driver_path = os.getcwd() + '/' + 'functions/raspberry_pi_IMU6050_driver/demo_dmp'
pipe_path   = "/tmp/gescon"

################# main #######################
# 1. record labelled dataset and get path to the raw_data folder.
path = record_labelled_dataset(frame_size, driver_path, pipe_path)

# 2. load data (X_raw) and labels (y)
X_raw = pd.read_csv(path + '/' + 'raw_data.csv', index_col = 0).values
y     = pd.read_csv(path + '/' + 'label.csv'   , header = None).values.ravel()
print('-'*30)
print('training classifier, please wait. This might take a couple of minutes.')

# preprocess raw data
X = preprocess_raw_dataset(X_raw, frame_size)
print('preprocessing completed')

# train classifier
print('learning .....')
clf_multi = train_clf(X, y)

# export classifier
joblib.dump(clf_multi, path + '/' + 'Multi_clf' + '.pkl')

# export path
with open (os.getcwd() + '/data/' + "clf_path.txt", "w") as path_file:
    path_file.write(path)

print('-'*30)
print('training completed - midi-chlorians levels rising; may the force be with you young Padawan')
print('-'*30)
