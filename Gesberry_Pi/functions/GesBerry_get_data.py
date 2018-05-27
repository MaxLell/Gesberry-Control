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

########################## imports ###########################
import os
import sys
import struct
import timeit
import time

import numpy as np
import subprocess

########################## sensor-functions #########################
def get_sensor_data(fifo, frame_size): #tested
    '''(fifo, int) --> np.array
    returns the raw data extracted from a fifo - pipe in form of a matrix.
    
    example:
    returns a vector with the shape of (1,7 *140) containing the linear accelerations of x,y,z
    and the quarternions qs,qx,qy,qz
    
    args:
        * fifo (fifo-object): A pipe containing the sensor readings
        * frame_size (int)  : Size of the time window of the recording
        
    returns:
        * raw_data (np.array): A Numpy.Array containing the sensor readings in a float-number
                               representation
    '''

    format = 'f' * 7 * (frame_size)
    bytes = fifo.read(struct.calcsize(format))
    raw_data = struct.unpack(format, bytes)
    raw_data = np.asarray(raw_data)
    return raw_data

def clear_pipe(pipe_path):
    '''(str) --> None
    clears the data from the data buffer, by reading and dumping it until,
    it needs to wait for new data input
    
    args:
        * pipe_path (str): String containing the path to the pipe
    
    returns:
        * None
    '''
    form = 'f' * 7 * 1 # c.a. time = 0.00990 ms
    time = 0
    while time < 0.008:
        start_time = timeit.default_timer()
        with open(pipe_path, 'rb') as fifo:
            dump = fifo.read(struct.calcsize(form))
        end_time = timeit.default_timer()
        time = end_time - start_time

def calibrate_sensor(pipe_path):
    '''(str) --> None
    it takes 20 seconds to calibrate the sensor at the beginning. 
    This step is required to initialize the sensor
    
    args:
        * pipe_path (str): String containing the path to the pipe
    
    returns:
        * None
    '''
    
    print('calibrating sensor - please wait for 20 Seconds', '\n'+'---------------------')
    for i in range(20):
        print(20 - i, 'seconds remaining')
        clear_pipe(pipe_path) # clear fifo to prevent sensor from fifo overflow
        time.sleep(1)
    print('---------------------', '\n'+'sensor calibrated')
    
def init_IMU(driver_path):
    '''
    (str) -> None
    This function starts the sensor driver as a subprocess
    
    args:
        * driver_path (str): String of the path to the sensor-driver.cpp file
    
    returns:
        * None
    '''
    
    # start c++ sensor driver
    subprocess.Popen([driver_path])
    print('sensor started')

def init_fifo(pipe_path):
    '''
    (str) --> fifo
    opens and returns the fifo
    
    args:
        * pipe_path (str): String of the path to the pipe
        
    returns:
        * fifo (fifo-object)
    '''
    
    fifo = open(pipe_path, "rb")
    print('read from pipe')
    return fifo

########################## recorded-datasets-functions #########################

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

    
    
    
    
    
    
    
    
    
    
