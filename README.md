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

A sample wav file has been include in which the background noise has
been obtained from [1] and the speech from [2].

#### References

[1] Koenig, M. (2018). Street Sounds | Effects | Sound Bites | Sound Clips from SoundBible.com. [online] Soundbible.com. Available at: http://soundbible.com/2175-Street.html [Accessed 14 Jun. 2018].

[2] Fromtexttospeech.com. (2018). From Text To Speech - Free online TTS service. [online] Available at: http://www.fromtexttospeech.com/ [Accessed 14 Jun. 2018].
