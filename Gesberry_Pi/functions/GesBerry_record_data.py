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

################# imports ##################
import numpy as np
import pandas as pd

import pathlib
import time
import re
import os
import sys

from GesBerry_get_data import get_sensor_data, clear_pipe, calibrate_sensor, init_IMU, init_fifo

################# functions ################

def get_labels_from_input(label): # tested
    '''(str) -> list of strings
    returns a list of strings, which represent the gesture's names

    example:
    >>> get_labels_from_input('swipe_right, swipe_left'):
    ['swipe_right', 'swipe_left']
    
    args:
        * label (str): A string containing all labels, separated by a comma
    
    returns:
        * list[str]: A list of strings, which contains all labels from the input string,
                     but now as separate strings in a list.
    '''

    return re.findall(r'\w+', label)

def wait_for_Enter_pressed(string):
    '''(str) --> None

    This function waits for an confirmation via pressing »Enter«
    '''

    try:
        input(string)
    except SyntaxError:
        pass


################ main program ################
def record_labelled_dataset(frame_size, driver_path, pipe_path):
    '''
    (int, str, str) --> str

    records a new labelled dataset and returns the folder path to where the dataset was created.
    X (IMU data) is stored in raw_data.csv and y (label) is stored in label.csv
    
    args:
        * frame_size  (int): Time duration of the recording window, represented by an integer
        * driver_path (str): String containing the path to the sensor-driver.cpp file
        * pipe_path   (str): String containing the path to the pipe file
        
    returns:
        * data_path   (str): String containing the path to the data folder
    '''
    # enter User's name
    identifier = input("pls enter User's name: ") + time.strftime('%d%m%y') # tested
    print('---------------------')

    # start sensor
    init_IMU(driver_path)

    # wait 1 sec
    time.sleep(1)

    # open pipe
    fifo = init_fifo(pipe_path)

    # skip calibration time
    calibrate_sensor(pipe_path)

    # get labels
    print('---------------------')
    label_list = get_labels_from_input(input('Please type in all gestures to record, separated by a comma :'))
    print('---------------------')

    # create folder
    pathlib.Path('data/' + identifier).mkdir(parents = True)

    # set datapath
    data_path = os.getcwd() + '/data/' + identifier + '/'

    # record gestures
    X = np.array([])
    y = np.array([])
    
    countdown = len(label_list) * 10 + 1 
    for h in range(0,10):
        for i in label_list:
            
            # subtract the countdown by 1 - indicates the remaining gestures to record
            countdown = countdown - 1
            
            # wait for keyboard input
            wait_for_Enter_pressed('press »Enter« to record gesture ---> '+  i + '\n')
            
            # clear pipe
            clear_pipe(pipe_path)

            # sensor writes current data into the fifo --> the only data in the fifo is now gesture information
            time.sleep(frame_size / 100) # wait for a period of time (the sensor subprocess writes the data to record into the pipe.

            #------- data ------
            x = get_sensor_data(fifo, frame_size) # read values from pipe

            x = np.array([x])
            if X.shape[0] == 0:
                X = x
            else:
                X = np.concatenate((X,x), axis=0) # stack all Records into a Matrix

            #------ label -------
            y = np.append(y, i) # stack all labels in the same order as the data

            #status update
            print('---------------------')
            print('record gesture »'+ i + '« completed')
            #print('remaining: ', countdown
            print('---------------------')

    #save data
    X_df = pd.DataFrame(X)
    X_df.to_csv(data_path + "raw_data.csv")

    #save labels
    np.savetxt(data_path + 'label' + ".csv",  y, delimiter=",", fmt="%s")

    # status update
    print('Gesture recording successfully finished')
    print('---------------------')

    # close pipe - subprocess gets terminated automatically
    fifo.close()

    # return the current path
    return data_path
