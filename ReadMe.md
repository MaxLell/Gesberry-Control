# Gesberry-Control

Overview
========

Gesture-recognition-system based on an IMU, a Raspberry Pi and Machine Learning. First the software needs to learn the gesture movements. To do so a human operator must train the system with his / her gestures. Afterwards the system is capable to detect gestures.
For that the IMU is mounted on the backside of the hand (not important wheather right or left),
attached to the Raspberry Pi (executes the SensorDriver and the Machine Learning), which then
predicts the gestures.The system needs to be trained with at least 10 gestures. 
When a trained operator uses the system the system acchieves a 10 fold cross validation score of
97%

This package is separated into two major folders: 
1. Raspberry Pi Package: This package includes:
  - A simple instruction of how to set up the system
  - The Machine Learning Algorithm for learning and interpreting gestures
  - The Sensor Driver (copied and modified from .... )
  - The User Interface, that gathers all functionality into few command line instructions that can be
    executed on the Raspberry Pi console.

2. Machine Learning Package: This package includes:
  - The Machine Learning Algorithm for learning and interpreting gestures
  - The performance Evaluation Tools (Confusion Matrix, Accuracy Score, 10 Fold Crossvalidation Score)
  - Documentation of several experiments (Data is included)
  - Designprocedure and Lessons learned

Dependencies
========
__Installation guide GesBerry-Pi__
1. Download and install Raspian (see https://www.raspberrypi.org/downloads/)
2. Activate SSH on the Raspberry Pi (see https://www.raspberrypi.org/documentation/remote-access/ssh/)
3. Change RaspberryPi default password to a password of your choice. 
4. open Terminal on the Raspberry Pi and enter the following lines (this updates the Raspberry Pi):
	1. `sudo apt-get update`
	2. `sudo apt-get upgrade`
5. Install python packages (needed for script-execution) - enter in the RaspBerry Pi Terminal:
	1. Install Numpy: `pip3 install numpy`
	2. Install Atlas: `sudo apt-get install libatlas-base-dev`
	2. Install SciPy: `sudo apt-get install python3-scipy`
	3. Install Scikit-Learn: `pip3 install -U scikit-learn`
	4. Install Pandas: `pip3 install pandas`
	5. Install libgtkmm-3.0-dev: `sudo apt-get install libgtkmm-3.0-dev`
6. Download the current GesBerry_Pi Code: Open Terminal on the Raspberry Pi and enter:
`git clone https://github.com/MaxLell/Gesberry-Control.git` The Code is downloaded to /home/pi directory
7. Navigate to the Raspberry-Pi-driver folder by entering `cd /home/pi/Gesberry-Control/Gesberry_Pi/Raspberry_pi_IMU6050_driver`
8. enter in Raspberry Pi Terminal: `make -j4` (compiles the IMU-sensor driver)


Usage
========
1. __learn gestures__: open Terminal, navigate to the /GesBerry_Pi folder and start the training procedure for the Gesture recognition system by entering `python3 GCI_train.py`. The script will guide you through the process. You can either create your own new gestures or you can use the ones that are provided (see `/GesBerry_MachineLearning/img/gestureTable_1.png`)

2. __detect gestures__: open Terminal, navigate to the /GesBerry_Pi folder and start the training procedure for the Gesture recognition system by entering `python3 GCI_classify.py`. The script will guide you through the process. 



Acknowledgements
==========

This research was conducted with the help of Univ.-Prof. Dr.-Ing. Thomas Ußmüller (University Innsbruck) and Dr. Gernot Grömer (Austrian Space Forum). 
The sensor driver for the Raspberry Pi was developped by Jeff Rowberg https://github.com/jrowberg/i2cdevlib and Richard Ghirst https://github.com/richardghirst/PiBits/tree/master/MPU6050-Pi-Demo. Minor changes were applied here in this research.

