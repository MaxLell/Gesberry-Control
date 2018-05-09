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

################ description: #############
# this script is called for signal preprocessing. 
# The gesture control system requires preoprecessed criterias
# with which the classification algorithm can distinguish 
# between the gestures and between no gesture / gesture.
# in this script all the necessary steps for 
# preprocessing are implemented.

############## imports ###################
import numpy as np
import pandas as pd

from scipy import signal
from scipy.fftpack import fft

from sklearn.preprocessing import StandardScaler

############### functions: #############

############################################################
############ time-progression-algorithm ####################
############################################################

def dyn_threshold(alpha, s):
    '''
    (float, np.array) -> number

    returns the threshold for segmenting the gesture 
    out of an input datastream. Only the signals high frequency
    components pass through this filter.
    There need to be other threshold values for alpha 
    than originally intended, since the acceleration scale a
    factor 200 higher than in the original paper
    '''

    return alpha * np.sqrt(np.var(s))

def find_flank_left(s, threshold):
    '''
    (np.array) --> np.array

    returns the start indizes of each gesture in a signal
    '''

    t = 0
    while s[t] < threshold:
        t = t + 1
    return t

def find_flank_right(s, threshold):
    '''
    (np.array) --> np.array

    returns the end indizes of each gesture in a signal
    '''

    t = len(s)-1
    indizes = np.array([])
    while s[t] < threshold:
        t = t - 1
    return t

def interpolate(y, size):
    '''
    (int, int, np.array) --> np.array

    Interpolates the clipped signal back to 
    a width of 100. It makes the signals more
    comparable
    '''

    x = np.linspace(0,len(y),len(y))
    xvals = np.linspace(0,len(y),size)
    yinterp = np.interp(xvals, x, y)
    return yinterp

def vari(s, window):
    '''
    (np.array, int) --> np.array

    returns the variance - vector of an input vector, 
    with a symetrical scan around the current index.
    the scan size in the minus and the plus direction is leni/2
    '''
    varianza = np.array([])
    for i in range(0,len(s)-window):
        varianza = np.append(varianza, np.var(s[i:i+window]))

    varianza = np.append(varianza, np.zeros(window))

    return varianza

def signal_var(s, window):
    '''
    (np.array, int) --> np.array

    filters the signal (vector) twice with a variance filter. 
    First it filters the signal with ascending indizes,
    for the second run it filters the signal reverse. 
    In the last step both values get added. By doing so,
    also the signal edges (the start and the end) get processed.
    '''

    fwd = vari(s,window)
    bwd = vari(s[::-1], window)
    c = fwd + bwd[::-1]
    return c

def running_mean(x):
    '''(np.array) --> np.array

    returns a compressed signal (compressrate 66%)
    '''
    q = np.array([])
    for i in range(2,len(x)-2,3):
         q = np.append(q,np.sum(x[i-2:(i+2)]))
    return q

def create_features_time_prog(s, frame_size):
    '''(np.array[1,980], int) --> np.array[1, 497]

    returns the gesture Essence of the inputfile. 
    In the input data the gesture may not be represented in
    the center of window. This function clips the relevant 
    gesture information out of the sequence interpolates
    it, and compresses it in the last step to reduce 
    computational load during classification.
    '''

    # reshape vector back to matrix representation
    x_raw = s.reshape(frame_size,7)
    
    # standard scale data
    scaler = StandardScaler()
    x_raw = scaler.fit_transform(x_raw)
    
    # segmenting the signal
    time_features = np.array([])
    for i in range(0,7):

        # i<3:  Acceleration
        # i>=3: Gyroscopes
        # Acceleration and Gyroscope's parameters need to be 
        # treated seperatly, since the gyroscopes scale 
        # different in variance and therefore
        # they require another threshold.

        if i < 3:
            var_size, thresh_factor =  10,  1.0 
        elif i >= 3:
            var_size, thresh_factor =  40, 1.5 

        # 1 vectorize x_raw
        s = x_raw[:,i]

        # 2 make absolute value
        s = np.abs(s)

        # 3 create variance filtered signal
        s = signal_var(s, var_size)

        # 4 Threshold - param
        threshold = dyn_threshold (thresh_factor, s)

        # 5 find left flank, find right flank
        start = find_flank_left (s, threshold) # right flank of the signal
        end   = find_flank_right(s, threshold) # left  flank of the signal

        # 6 cut signal
        s = x_raw[start:end, i]

        # 7 interpolate to 140
        s = interpolate(s, size=frame_size)

        # 8 compress signal by 66%
        s = running_mean(s)

        # 9 append to feature vector: interpolated signal
        time_features = np.append(time_features, s)

    # FFT - Features
    fft_features = fourier_spec_signal(x_raw, frame_size)[:25,:].reshape(1,-1)

    # stack feature vector
    features = np.array([])
    features = np.append(features, time_features)
    features = np.append(features, fft_features)
    return features.reshape(1,-1)

################################################
############## fixed features ##################
################################################



#################### filter #################

def bandpass_filter(y, f_cutoff_low, f_cutoff_high, f_abt):
    
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
    
    nyq = 0.5 * f_abt
    normal_cutoff = f_cutoff / nyq
    
    lowpass_signal = np.zeros(y.shape)
    for i in range(y.shape[1]):
        b, a = signal.butter(5, normal_cutoff , 'low') # Tiefpass
        lowpass_signal[:,i] = signal.filtfilt(b, a, y[:,i])
    return lowpass_signal


########### feature functions for calculation ###########


def ACSkew(x):
    '''(np.vector) --> float
    
    returns a measure of the peakedness of the accelerometer / 
    gyroscope signal over a given window
    '''
    n = len(x)
    
    sum_numerator   = np.sum (np.power ((x - np.mean(x)), 3))
    sum_denominator = np.sum (np.power ((x - np.mean(x)), 2))

    return (np.sqrt(n) * sum_numerator) / (np.power(sum_denominator, 3/2))

def ACKur(x):
    '''(np.vector) --> float
    
    returns a measure of the peakedness of the accelerometer / gyroscope 
    signal over a given window or a measure of its relative flatness 
    as compared to the normal distribution.
    '''
    n = len(x)
    
    sum_numerator   = np.sum (np.power ((x - np.mean(x)), 4))
    sum_denominator = np.sum (np.power ((x - np.mean(x)), 2))

    return (n * sum_numerator / (np.power(sum_denominator,2)))-3

def calc_signal_energy(y, number_of_samples = None):
    '''(np.array[140,7], [int]) --> np.array[30,7], np.array[40,7]
    
    returns the signal energy of the accelerometers and gyroscopes.
    '''
    if number_of_samples == None:
        N = 140
    
    yf = fft(y)
    half_abs_yf = 2/N*np.abs(yf[0:N//2,:])
    acl_energy = np.sum(half_abs_yf[:,0:3])
    gy_energy = np.sum(half_abs_yf[:,3:7])
    return acl_energy, gy_energy

def fourier_spec_axis(y):
    '''(np.array) -> np.array
    calculates the fft spectrum of the signal and returns it.
    '''
    N = len(y)
    yf = fft(y)
    half_abs_yf = 2/N*np.abs(yf[0:N//2])
    return half_abs_yf

def fourier_spec_signal(y, frame_size= None):
    '''(np.array, int) --> np.array

    returns the standard scaled fourierspectrum (70,7) 
    of each vector (140,1) of the input signal matrix (140,7)
    '''
    if frame_size == None:
        frame_size = 140
    
    fourier_spec = np.array([])
    for i in range(y.shape[1]):
        fourier_spec = np.append(fourier_spec, 
                                 fourier_spec_axis(y[:, i]))
    fourier_spec = fourier_spec.reshape(frame_size//2, 7)
    scaler = StandardScaler()
    fourier_spec = scaler.fit_transform(fourier_spec)
    return fourier_spec

def varianz(x):
    '''(np.array) -> np.array [1, 5]
    '''
    a = np.var(x)
    s = np.array([])
    
    one_fourth = (len(x)//4)
    
    for i in range(0,len(x), one_fourth):
        s = np.append(s, np.var(x[i:i+one_fourth]))
    s = np.append(s, a)
    return s


def ACVar(y):
    '''(np.array[140,7]) --> np.array
    
    returns the signal's variance
    '''
    a = np.array([])
    for i in range(y.shape[1]):
        a = np.append(a, varianz(y[:,i]))
    return a


##################### main ##############################

def create_features_fixed(y, frame_size, f_abt = None):
    '''(np.array[1,140], int, [int] --> np.array[1,259]
    
    returns a variety of features out of the input signal
    returns the feature's names
    '''
    
    if f_abt == None:
        f_abt = 100
        
    # reshape vector back to matrix representation
    y = y.reshape(frame_size,7)
        
    ###################### 1 ######################
    # LowPass Filter 1Hz
    y_lowpass_filter = lowpass_filter(y, 1, f_abt)

    # DCMean
    acl_DCMean_x    = np.mean(y_lowpass_filter[:,0])
    acl_DCMean_y    = np.mean(y_lowpass_filter[:,1])
    acl_DCMean_z    = np.mean(y_lowpass_filter[:,2])
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

    ################### 2 ##################

    # BandPass Filter 1Hz - 20Hz
    y_bandpass_filter = bandpass_filter(y, 0.1, 25, f_abt) 
    # modifikation der Filterparameter

    # ACAbsMean
    acl_ACAbsMean_x    = np.mean(np.abs(y_bandpass_filter[:,0]))
    acl_ACAbsMean_y    = np.mean(np.abs(y_bandpass_filter[:,1]))
    acl_ACAbsMean_z    = np.mean(np.abs(y_bandpass_filter[:,2]))
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
    
    

    ################# 3 #################

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

    ################ 4 ###################
    
    # calculate fouier spectrum
    acl_gy_fourier_spectrum = fourier_spec_signal(
        y, frame_size)[:30,:].reshape(1,-1)
    
    
    ############### stack to vector #######
    
    ################# 1,2,3 ###############
    
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
    
    ################### own creeations ##################
    
    features = np.append(features, acl_gy_fourier_spectrum)                    
    return features.reshape(1,-1)

def preprocess_raw_dataset(X_raw, frame_size, mode = None):
    '''(np.array, int, [str]) --> np.array
    Returns the preprocessed dataset (X), which was transformed via the
    create_features-routine.
    The mode defines the preprocessing routine to use. In its standard
    configuration the fixed features are used.
    '''
    if mode == None:
        mode = 'fixed'
    elif mode == 'time':
        mode = 'time'
    
    X = np.array([])
    for row in range(X_raw.shape[0]):
        x_raw = X_raw[row, :]
        
        if mode == 'fixed':
            features = create_features_fixed(x_raw, frame_size) # create features fixed
        else:
            features = create_features_time_prog(x_raw, frame_size)  #create features time-progressive
        
        if X.shape[0] == 0:
            X = features
        else:
            X = np.concatenate((X,features), axis=0) # stack X to matrix

    return X
