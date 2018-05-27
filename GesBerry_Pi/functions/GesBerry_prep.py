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

# imports
import numpy as np # -------------------------------------- vers. numpy:  1.14.0
import pandas as pd # ------------------------------------- vers. pandas: 0.22.0

from scipy import signal # -------------------------------- vers. scipy: 1.0.0
from scipy.fftpack import fft
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer

#################### main function #####################################

def preprocess_raw_data(X_raw, frame_size):
    """(np.array, int) --> np.array
    
    Function that transforms the raw time progression signal into feature vectors. Afterwards
    the feature vector is normalized.
    (Dimensions of matrizes depend from the frame_size - the dimensions
    that are listed here refer to a frame_size of 140! Same comes for all
    listed subfunctions)
    
    n = 130
    m = 980
    
    args:
        * X_raw (np.array[n,m]): raw data - can be a single vector or a matrix
        * frame_size (int): recording window size
    
    Returns:
        * Preprocessed dataset (X) - vector or matrix
    
    TODO:
        * Test efficiency of features to streamline code
            - Only FFT-Features
            - Only Tapia features
            - Selected Tapia Features
        * standarize the choice of variables in functions: 
          create_feature_vector: y = x_raw: confusing description.
        * rework comments of helper functions
        * optimize Code for matrix process: Currently many methods perform the same task
          of splitting up the matrix into a feature vector --> streamline code
        * put features in a dictinary instead of a list - better quality management of 
          the features possible
          e.g.: 
          axis = [ax,ay,az,qs,qx,qy,qz]
          for in range(0,len(axis)):
              feature_vector['acl_DCMean_'+axis[i]] = np.mean(y_lowpass_filter[:,i])
        * debug and check functions:
            - rework signal energy functions
            - rework fft signals
        * plot feature importance
        * place features into python dictionaries
             
    """
    norm = Normalizer()
    X = np.array([])
    for row in range(X_raw.shape[0]): 
        
        # in case only one vector (X_raw) is provided: Output normalized featurevector
        # in case X_raw represents a matrix: Split matrix up into vectors, calculate feature
        # vectors, stack them and output the transformed dataset X
        
        # split up matrix into vector and process single vectors
        x_raw = X_raw[row, :]
        
        # create feature vector from raw data vector
        feature_vector = create_feature_vector(x_raw, frame_size) 
    
        # normalize featurevector
        feature_vector = norm.fit_transform(feature_vector) 
        
        # in case only one vector (X_raw) is provided: Output featurevector
        if X.shape[0] == 0:
            X = feature_vector
        
        # in case a Matrix (X_raw) is provided: Stack to matrix and output matrix
        else:
            X = np.concatenate((X,feature_vector), axis=0) # --------- 

    return X

########################################################################

def create_feature_vector(y, frame_size, f_abt = 100):
    """(np.array[1,m], int, [int] --> np.array[1,x]
    
    Transforms the raw time progression signal vector into feature vector.
    
    args:
        * y (np.array[]):      raw signal
        * frame_size (int):    duration of recording window in 100ms 
                               (e.g.: frame_size = 140 -> record time window: 1.4 sec)
    
    returns:
        * features (np.array[1,x]): feature vector

    """ 
    # reshape vector back to matrix representation
    y = y.reshape(frame_size,7)
    
    ############################ Tapia Features #########################
    ###################### 1. Measures of body posture #################
    # LowPass Filter 1Hz
    y_lowpass_filter = lowpass_filter(y, 1, f_abt)

    # DCMean
    acl_DCMean_x    = np.mean(y_lowpass_filter[:,0]) #ax
    acl_DCMean_y    = np.mean(y_lowpass_filter[:,1]) #ay
    acl_DCMean_z    = np.mean(y_lowpass_filter[:,2]) #az
    gy_DCMean_qs    = np.mean(y_lowpass_filter[:,3]) #qs
    gy_DCMean_qx    = np.mean(y_lowpass_filter[:,4]) #qx
    gy_DCMean_qy    = np.mean(y_lowpass_filter[:,5]) #qy
    gy_DCMean_qz    = np.mean(y_lowpass_filter[:,6]) #qz

    # DCTotalMean
    acl_DCTotalMean = np.sum(acl_DCMean_x + acl_DCMean_y + 
                             acl_DCMean_z)
    gy_DCTotalMean  = np.sum(gy_DCMean_qs + gy_DCMean_qx + 
                             gy_DCMean_qy + gy_DCMean_qz)

    # DCArea
    acl_DCArea = np.sum(y[:,0] + y[:,1] + y[:,2])
    gy_DCArea  = np.sum(y[:,3] + y[:,4] + y[:,5] + y[:,6])

    # DCPostureDist
    acl_DC_PostureDist_XZ       = acl_DCMean_x - acl_DCMean_z #X-Z
    acl_DC_PostureDist_XY       = acl_DCMean_x - acl_DCMean_y #X-Y
    acl_DC_PostureDist_YZ       = acl_DCMean_y - acl_DCMean_z #Y-Z
    gy_DC_PostureDist_qsqx      = gy_DCMean_qs - gy_DCMean_qx #qs-qx
    gy_DC_PostureDist_qxqy      = gy_DCMean_qx - gy_DCMean_qy #qx-qy
    gy_DC_PostureDist_qyqz      = gy_DCMean_qy - gy_DCMean_qz #qy-qz
    gy_DC_PostureDist_qzqs      = gy_DCMean_qz - gy_DCMean_qs #qz-qs

    ################### 2. Measures of motion shape ##################

    # BandPass Filter 1Hz - 20Hz
    y_bandpass_filter = bandpass_filter(y, 0.1, 25, f_abt) 

    # ACAbsMean
    acl_ACAbsMean_x    = np.mean(np.abs(y_bandpass_filter[:,0])) #ax
    acl_ACAbsMean_y    = np.mean(np.abs(y_bandpass_filter[:,1])) #ay
    acl_ACAbsMean_z    = np.mean(np.abs(y_bandpass_filter[:,2])) #az
    gy_ACAbsMean_qs    = np.mean(np.abs(y_bandpass_filter[:,3])) #qs
    gy_ACAbsMean_qx    = np.mean(np.abs(y_bandpass_filter[:,4])) #qx
    gy_ACAbsMean_qy    = np.mean(np.abs(y_bandpass_filter[:,5])) #qy
    gy_ACAbsMean_qz    = np.mean(np.abs(y_bandpass_filter[:,6])) #qz

    # ACAbsArea
    acl_ACAbsArea = np.sum(np.abs(y_bandpass_filter[:,0]) + 
                           np.abs(y_bandpass_filter[:,1]) + 
                           np.abs(y_bandpass_filter[:,2]))
    gy_ACAbsArea  = np.sum(np.abs(y_bandpass_filter[:,3]) + 
                           np.abs(y_bandpass_filter[:,4]) + 
                           np.abs(y_bandpass_filter[:,5]) +
                           np.abs(y_bandpass_filter[:,6]))

    # ACSkew
    acl_ACSkew_x    = ACSkew(y_bandpass_filter[:,0])
    acl_ACSkew_y    = ACSkew(y_bandpass_filter[:,1])
    acl_ACSkew_z    = ACSkew(y_bandpass_filter[:,2])
    gy_ACSkew_qs    = ACSkew(y_bandpass_filter[:,3])
    gy_ACSkew_qx    = ACSkew(y_bandpass_filter[:,4])
    gy_ACSkew_qy    = ACSkew(y_bandpass_filter[:,5])
    gy_ACSkew_qz    = ACSkew(y_bandpass_filter[:,6])

    # ACKur
    acl_ACKur_x = ACKur(y_bandpass_filter[:,0])
    acl_ACKur_y = ACKur(y_bandpass_filter[:,1])
    acl_ACKur_z = ACKur(y_bandpass_filter[:,2])
    gy_ACKur_qs = ACKur(y_bandpass_filter[:,3])
    gy_ACKur_qx = ACKur(y_bandpass_filter[:,4])
    gy_ACKur_qy = ACKur(y_bandpass_filter[:,5])
    gy_ACKur_qz = ACKur(y_bandpass_filter[:,6])

    ################# 3. Measures of motion energy #################

    #ACVar
    acl_gy_ACVar = ACVar(y_bandpass_filter)
    acl_gy_ACAbsCV = ((np.sqrt(ACVar(y_bandpass_filter)))
                      /np.mean(y_bandpass_filter))*100
    
    # ACEnergy
    acl_ACEnergy, gy_ACEnergy = calc_signal_energy(y_bandpass_filter)

    # ACBandEnergy
    acl_ACBandEnergy, gy_ACBandEnergy = calc_signal_energy(
        bandpass_filter(y, 0.3, 3.5, f_abt)) 

    #ACLowEnergy
    acl_ACLowEnergy, gy_ACLowEnergy = calc_signal_energy(
        bandpass_filter(y, 0.0001, 0.71, f_abt))

    #ACModVigEnergy
    acl_ACModVigEnergy, gy_ACModVigEnergy = calc_signal_energy(
        bandpass_filter(y, 0.71, 10, f_abt)) 
    
    #ACGesSpec1Energy
    acl_ACGesSpec1Energy, gy_ACGesSpec1Energy = calc_signal_energy(
        bandpass_filter(y, 0.1, 15, f_abt)) 
    
    #ACGesSpec2Energy
    acl_ACGesSpec2Energy, gy_ACGesSpec2Energy = calc_signal_energy(
        bandpass_filter(y, 3, 20, f_abt)) 
    
    #ACGesSpec3Energy
    acl_ACGesSpec3Energy, gy_ACGesSpec3Energy = calc_signal_energy(
        bandpass_filter(y, 10, 25, f_abt)) 

    ################# Stack to vector ####################
    
    features = np.array([])
    features = np.append(features,
                         [acl_DCMean_x, acl_DCMean_y, acl_DCMean_z, 
                          gy_DCMean_qs, gy_DCMean_qx, gy_DCMean_qy, 
                          gy_DCMean_qz,
                          acl_DCTotalMean, gy_DCTotalMean, 
                          acl_DCArea, gy_DCArea,
                          acl_DC_PostureDist_XZ, acl_DC_PostureDist_XY, 
                          acl_DC_PostureDist_YZ,
                          gy_DC_PostureDist_qsqx, gy_DC_PostureDist_qxqy, 
                          gy_DC_PostureDist_qyqz, gy_DC_PostureDist_qzqs,
                          acl_ACAbsMean_x, acl_ACAbsMean_y, acl_ACAbsMean_z, 
                          gy_ACAbsMean_qs, gy_ACAbsMean_qx, gy_ACAbsMean_qy, 
                          gy_ACAbsMean_qz,
                          acl_ACAbsArea, gy_ACAbsArea,
                          acl_ACSkew_x, acl_ACSkew_y, acl_ACSkew_z, 
                          gy_ACSkew_qs, gy_ACSkew_qx, gy_ACSkew_qy, 
                          gy_ACSkew_qz,
                          acl_ACKur_x, acl_ACKur_y, acl_ACKur_z, 
                          gy_ACKur_qs, gy_ACKur_qx, gy_ACKur_qy, gy_ACKur_qz,
                          acl_ACEnergy, gy_ACEnergy, acl_ACBandEnergy, 
                          gy_ACBandEnergy,
                          acl_ACLowEnergy, gy_ACLowEnergy, 
                          acl_ACModVigEnergy, gy_ACModVigEnergy,
                          acl_ACGesSpec1Energy, gy_ACGesSpec1Energy,
                          acl_ACGesSpec2Energy, gy_ACGesSpec2Energy,
                          acl_ACGesSpec3Energy, gy_ACGesSpec3Energy])
    
    
    ########################### FFT - Features #############################
    ###################### calculate fouier spectrum #######################
    
    acl_gy_fourier_spectrum = fourier_spec_signal(y, frame_size)[:30,:].reshape(1,-1)
    features = np.append(features, acl_gy_fourier_spectrum)                    
    
    ######################### return feature vector ########################
    
    return features.reshape(1,-1)

    #################### filter helper functions #################

def bandpass_filter(y, f_cutoff_low, f_cutoff_high, f_abt):
    """(np.array, float, loat, int) -> np.array
    
    Returns a bandpass filtered signal
    
    args:
        * y (np.array) ---------- : raw signal
        * f_cutoff_low (float) -- : lower cutoff frequency
        * f_cutoff_high (float) - : high cutoff frequency
        * f_abt (int) ----------- : sample frequency
    
    returns:
        * bandpass_signal (np.array) : band-pass filtered signal
    """
    nyq = 0.5 * f_abt
    low_cutoff = f_cutoff_low / nyq
    high_cutoff = f_cutoff_high / nyq
    
    bandpass_signal = np.zeros(y.shape)

    for i in range(y.shape[1]):
            b, a = signal.butter(5, [low_cutoff, high_cutoff] , 'bandpass') 
            # Bandpass
            bandpass_signal[:,i] = signal.filtfilt(b, a, y[:,i])
    
    return bandpass_signal

def lowpass_filter(y, f_cutoff, f_abt):
    """(np.array, float, int) -> np.array
    
    Returns a lowpass filtered signal
    
    args:
        * y (np.array) ---- : raw signal
        * f_cutoff(float) - : cutoff frequency
        * f_abt (int) ----- : sample frquency
    
    returns:
        * lowpass_signal (np.array) = low-pass filtered signal
    """
    
    nyq = 0.5 * f_abt
    normal_cutoff = f_cutoff / nyq
    
    lowpass_signal = np.zeros(y.shape)
    for i in range(y.shape[1]):
        b, a = signal.butter(5, normal_cutoff , 'low') # Tiefpass
        lowpass_signal[:,i] = signal.filtfilt(b, a, y[:,i])
    
    return lowpass_signal


    ########### functions for feature calculations ###########


def ACSkew(x):
    """(np.vector) -> float
    
    returns a measure of the peakedness / skewness of the accelerometer / 
    gyroscope signal over a given window
    
    args:
        * x (np.array) : time-progression signal
    
    returns:
        * processed signal (np.array) : Skewness of the signal
    
    """
    n = len(x)
    
    sum_numerator   = np.sum (np.power ((x - np.mean(x)), 3))
    sum_denominator = np.sum (np.power ((x - np.mean(x)), 2))

    return (np.sqrt(n) * sum_numerator) / (np.power(sum_denominator, 3/2))

def ACKur(x):
    """(np.vector) -> float
    
    returns a measure of the peakedness of the accelerometer / gyroscope 
    signal over a given window or a measure of its relative flatness 
    as compared to the normal distribution.
    
    
    args:
        * x (np.array) : time-progression signal
    
    returns:
        * processed signal (np.array) : Kurtosis of the signal
    
    """
    n = len(x)
    
    sum_numerator   = np.sum (np.power ((x - np.mean(x)), 4))
    sum_denominator = np.sum (np.power ((x - np.mean(x)), 2))

    return (n * sum_numerator / (np.power(sum_denominator,2)))-3

def calc_signal_energy(y, frame_size = 140):
    """(np.array, int) -> np.array[30,7], np.array[40,7]
    
    returns the signal energy of the accelerometers and gyroscopes.
    
    args:
        * y (np.array) ----- : time-progression signal
        * frame_size (int) - : duration of recording window in 100ms 
    
    returns:
        * acl_energy (float) : signal energy of the acceleration signal
        * gy_energy  (float) : signal energy of the gyroscope's signal
    
    """
    
    yf = fft(y)
    half_abs_yf = 2/frame_size*np.abs(yf[0:frame_size//2,:])
    acl_energy = np.sum(half_abs_yf[:,0:3])
    gy_energy = np.sum(half_abs_yf[:,3:7])
    return acl_energy, gy_energy

def fourier_spec_axis(y):
    """(np.array) -> np.array
    
    calculates the fft spectrum of the signal and returns it.
    
    args:
        * y (np.array) : raw signal
    
    returns:
        * half_abs_yf (np.array) : half of the fft signal
    """
    N = len(y)
    yf = fft(y)
    half_abs_yf = 2/N*np.abs(yf[0:N//2])
    return half_abs_yf

def fourier_spec_signal(y, frame_size= 140):
    """(np.array, int) -> np.array

    returns the standard scaled fourierspectrum (70,7) 
    of each vector (140,1) of the input signal matrix (140,7)
    
    
    args:
        * y (np.array) = raw time progression signal (matrix)
        * frame_size = duration of recording window in 100ms 
        
    returns:
        * fourier_spec (np.array) = fourier spectrum
    """
    
    fourier_spec = np.array([])
    for i in range(y.shape[1]):
        fourier_spec = np.append(fourier_spec, 
                                 fourier_spec_axis(y[:, i]))
    fourier_spec = fourier_spec.reshape(frame_size//2, 7)
    scaler = StandardScaler()
    fourier_spec = scaler.fit_transform(fourier_spec)
    return fourier_spec


def ACVar(y):
    """(np.array) -> np.array
    
    returns the signal's variance
    
    args:
        * y (np.array) : time-progression signal
    
    returns:
        * a (np.array) : signal's variances
    
    """
    a = np.array([])
    for i in range(y.shape[1]):
        a = np.append(a, np.var(y[:,i]))
    return a
