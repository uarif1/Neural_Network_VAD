# Neural Network VAD

A script and a package to perform Speech/Non-speech identification (SNI) using
Neural Network on .wav files.

The function that performs SNI is in VAD.py and is called Neural_Network_VAD.
It returns as tuple the SNI results from evaluation on a Convolution, LSTM,
Dense Neural Network (0 index of tuple) and on a LSTM Dense Neural Network
(1st index of tuple).

VAD_script.py is a wrapper that will save the results of SNI as a plot in a .png
file and as a csv file.

**This has only been tested on and is only expected to work using
Python 3.6 [64 bit] due to tensorflow requirement.**
