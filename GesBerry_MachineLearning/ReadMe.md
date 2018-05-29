## Overview:
This section gives insight in the experiments that were run to evaluate the gesture recogntion process.

The Machine Learning software can be run independantly from the Raspberry Pi Hardware. In the conclusion part of the Jupyter notebook the major concepts for designing a new gesture set are proposed. The notebook also introduces the human impact factors in the gesture recognition process.

The provided notebook can be used to validate the existing solution or it can be used, when new gesture sets are developed. In the second case the test cases can easily be ported to a new gesture set. To do so copy the commands and function calls that are shown in the `Gesberry_Machine_learning.ipynb` notebook

There are two ways to run the evaluation juypter notebook:
1. Execute it on the Raspberry Pi
2. Execute it on an external machine.

First an __Installation guide__ for both options is provided. 

The __Usage__ of the sub-package is straight-forwards: Simply execute Jupyter notebook: `GesBerry_Machine_Learning.ipynb` in the jupyter notebook environment. 

## Installation Guide Option 1 (Raspberry Pi):
`sudo su -`
`apt-get update`
`apt-get upgrade`
`pip3 install --upgrade pip`
`apt-get install python3-matplotlib`
`apt-get install python3-scipy`
`apt-get install libatlas-base-dev`

`reboot`
`sudo pip3 install seaborn, pandas, jupyter`
`sudo pip3 install -U scikit-learn`
`sudo apt-get clean`

## Installation guide Option 2 (external Machine: Windows, Mac, Linux):
Install Anaconda (https://anaconda.org/). This includes all the required scripts by default or follow the procedure of Option 1.
