## Overview:
This section gives insight in the experiments that were run to evaluate the gesture recogntion process.

The Machine Learning software can be run independantly from the Raspberry Pi Hardware. In the conclusion part of the Jupyter notebook the major concepts for designing a new gesture set are proposed. The notebook  also introduces the human impact factors in the gesture recognition process. --> see `GesBerry_Machine_learning.ipynb`

The provided notebook can be used to validate the existing solution or it can be used, when new gesture sets are developed. In the second case the test cases can easily be ported to a new gesture set. To record data use the GesBerry_Pi platform or structure the data in the same way. --> see `GesBerry_CheckGestures.ipynb` 

The Jupyter notebooks can be run on either the Raspberry Pi or on an external machine. An __Installation guide__ is provided for both options.

The __Usage__ of the sub-package is straight-forwards: 
Open the Jupyter Notebook IDE and run the scripts.

## Installation Guide Option 1 (Raspberry Pi):
`sudo su -` </b>
`apt-get update` </b>
`apt-get upgrade` </b>
`pip3 install --upgrade pip` </b>
`apt-get install python3-matplotlib` </b>
`apt-get install python3-scipy` </b>
`apt-get install libatlas-base-dev` </b>

`reboot` </b>
`sudo pip3 install seaborn, pandas, jupyter` </b>
`sudo pip3 install -U scikit-learn` </b>
`sudo apt-get clean` </b>

## Installation guide Option 2 (external Machine: Windows, Mac, Linux):
Install Anaconda (https://anaconda.org/). This includes all the required scripts by default or follow the procedure of Option 1.
