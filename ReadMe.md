# Gesberry-Control
Gesture Control System based on an IMU, a Raspberry Pi and Machine Learning

For that the IMU is mounted on the backside of the hand (not important wheather right or left),
attached to the Raspberry Pi (executes the SensorDriver and the Machine Learning), which then
predicts the gestures. Nevertheless the system needs to be trained with at least 10 gestures. 
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

## Upcoming modifications:
- Enhance ReadMe.md:
    - work on better description
- Clean Sourcecode: 
	- Delete old code fragments
	- Streamline functions
    - Clean Sensor Driver from unneccessary functions
- Delete unneccessary data from different operators
- Setup an explanatory Jupyter Notebook that investigates the whole dataprocessing steps: 
    - Use better defined Data folder descriptors
    - Rework Performance measurement GCI.ipynb
- Write a complete setup and installation guide for the Raspberry Pi

