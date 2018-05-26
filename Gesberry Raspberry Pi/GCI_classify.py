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

########################## imports #########################
import numpy as np
import pandas as pd

import os
import sys
sys.path.append(os.getcwd() + '/' + 'functions')

import time
import timeit

from GesBerry_prep import preprocess_raw_data
from GesBerry_get_data import get_sensor_data, clear_pipe, calibrate_sensor, init_IMU, init_fifo

from sklearn.externals import joblib

########################## globals #########################
with open (os.getcwd() + '/data/' + 'clf_path.txt', "r") as pathfile:
    path = pathfile.read()

#path = path[:-1]

driver_path = os.getcwd() + '/' + 'functions/raspberry_pi_IMU6050_driver/demo_dmp'
pipe_path   = "/tmp/gescon"
frame_size  = 140

# load classifier
clf_multi  = joblib.load(path + '/' + 'Multi_clf' + '.pkl')
print('successfully loaded classifier')

########################## functions ########################
def wait_for_trigger(string):
    '''(str) --> None
    
    returns only when »Enter« is pressed
    '''
    
    try:
        input(string)
    except SyntaxError:
        pass

######################### main ##############################
# start sensor
init_IMU(driver_path)

# wait 1 sec
time.sleep(1)

# open pipe
fifo = init_fifo(pipe_path)

# skip calibration time
calibrate_sensor(pipe_path)

print('sensor set up - read from pipe')

while True:
    # await trigger signal - for the current scenario: press Enter on the keyboard
    wait_for_trigger('press »Enter« to recognize gesture') # Trigger!
    
    # clear pipe
    clear_pipe(pipe_path)
    
    # get new sensor values
    x_raw = get_sensor_data(fifo, frame_size)
    
    # create features from raw values
    x = preprocess_raw_data(x_raw, frame_size)
    
    # predict which gesture is represented in x_raw
    y_pred = clf.predict(x)
    
    # output the prediction
    print('Prediction: --->',y_pred,'\n'+'--------------------------------------')
    
fifo.close()
sys.exit()
