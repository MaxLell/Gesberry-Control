# Gesberry-Control

## Overview


This package contains Gesture-recognition-system based on an IMU6050 (Inertial Measurement Unit), a Raspberry Pi and Machine Learning. The following graph shows the system setup:
<img src="img/GesBerry_Pi.png">

The sensor is mounted on the backside of the middlefinger (see figure) and records a gesture after a trigger-button-press. The major processing unit is the Raspberry Pi. It executes:
- sensor driver
- data-processing
- Machine-Learning (learning from data and classifying new samples)
Finally the gestures are output via the console.

First the software needs to learn the gesture movements. To do so a human operator must train the system with his / her gestures. The system needs to be trained with at least 10 gestures. A trained operator achieves a 10-fold-cross-validation accuracy-score of 97%.

This package is separated into two folders: 
1. The __GesBerry_Pi Package__: This package contains the complete sourcecode to run the Gesture recognition software on a Raspberry Pi. Furthermore a complete installation guide is provided and a circuit diagram is included on how to physically set up the system.

2. The __GesBerry_MachineLearning Package__: This package contains an executable jupyter notebook, which contains several system experiments. On the one hand these show the current systems capabilities in terms of detection accuracy (also for future systems) and on the other hand it provides valueable information to maximize the detection accuracy. Here the impact of the human operator is shown.


## Dependencies and Installation Guides

* numpy
* scipy
* scikit-learn
* pandas
* Atlas

The installation guides for the concerning packages are included in the folders. Besides the software implentation also a circuit diagram for the hardware development is provided.

Acknowledgements
==========
The whole project was developed in a Masterthesis of Maximilian Lell at the University of Innsbruck (see MasterThesis)
This research was conducted with the help of Univ.-Prof. Dr.-Ing. Thomas Ußmüller (University Innsbruck) and Dr. Gernot Grömer (Austrian Space Forum). 
The sensor driver for the Raspberry Pi was developped by Jeff Rowberg https://github.com/jrowberg/i2cdevlib and Richard Ghirst https://github.com/richardghirst/PiBits/tree/master/MPU6050-Pi-Demo. The sensor driver is modified to write the sensor's output into a fifo-pipe.