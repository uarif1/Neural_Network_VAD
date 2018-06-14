# Neural Network VAD

**This has only been tested and is only expected to work using
Python 3.6 [64 bit] due to tensorflow requirement.**

A package to perform Speech/Non-speech Identification (SNI) using
Neural Networks on .wav files.

The function that performs SNI is in VAD.py and is called Neural_Network_VAD.
Provided the speech signal (as a 1-D array) and the sampling frequency, it returns as
a tuple the SNI results from prediction using a Convolution-LSTM-Dense
Neural Network (0 index of tuple) and a LSTM-Dense Neural Network
(1st index of tuple).

VAD_script.py is a wrapper that will save the results of SNI as a plot in a .png
file and as a csv file.

use: python VAD_script.py  sample.wav

for help: python VAD_script.py -h
