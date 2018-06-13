
import os

import matplotlib.pyplot as plt
import numpy as np

from scipy.signal import medfilt
from keras.models import load_model

from voicebox.fxpefac import fxpefac
from voicebox.melcepst import melcepst

TIME_PER_FRAME = 0.025
FILTER = False
DERIVATIVES = True


def derivatives(features):
    startframe = 10
    no_of_feats = features.shape[0]
    no_of_frames = features.shape[1]
    feat_der = np.zeros((3*no_of_feats, no_of_frames))
    feat_der[:no_of_feats, :no_of_frames] = features

    # first derivatives
    for k in range(no_of_feats):
        for l in range(startframe, no_of_frames):
            vec = features[k, l-startframe:l+1]
            val = 0
            for p in range(len(vec)):
                val += vec[-1]-vec[p]
            feat_der[k+no_of_feats, l] = val

    # second derivatives
    for k in range(no_of_feats):
        for l in range(startframe, no_of_frames):
            vec = feat_der[k+no_of_feats, l-startframe:l+1]
            val = 0
            for p in range(len(vec)):
                val += vec[-1]-vec[p]
            feat_der[k+2*no_of_feats, l] = val

    return feat_der


def Neural_Network_VAD(speech, fs, filename='speech', show_plt=True, plt_save=True,
                       excel=True):

    samples_per_frame = int(TIME_PER_FRAME*fs)
    fx, tx, pv, fv = fxpefac(speech, fs, TIME_PER_FRAME)
    pefac_start_offset = int(np.ceil(tx[0]*fs-samples_per_frame/2)+1)
    pefac_end_offset = int(np.ceil(tx[-1]*fs+samples_per_frame/2)+1)

    c, tc = melcepst(speech[pefac_start_offset-1:pefac_end_offset], fs,
                     n=samples_per_frame, inc=samples_per_frame)

    features = np.vstack((pv.T, np.log10(fx).T, c.T))
    if FILTER:
        features = medfilt(features, (1, 11))

    if DERIVATIVES:
        features = derivatives(features)

    model = load_model('Conv1D_LSTM_1_frame.h5')
    pred_conv_lstm = model.predict(features.T[np.newaxis])[0, :, 0].T

    model = load_model('LSTM_1_frame.h5')
    pred_lstm_only = model.predict(features.T[np.newaxis])[0, :, 0].T
    print(pred_lstm_only.shape)
    plt.subplot(3, 1, 1)
    plt.plot(np.arange(len(speech))/fs, speech, label='speech')
    plt.subplot(3, 1, 2)
    plt.plot(tx, pred_conv_lstm, label='Convolution_LSTM_Dense_Neural_Network')
    plt.subplot(3, 1, 3)
    plt.plot(tx, pred_lstm_only, label='LSTM_Dense Neural Network')
    plt.legend()
    import IPython
    IPython.embed()
    if show_plt:
        plt.show()
    if plt_save or excel:
        if not os.path.exists('results'):
            os.makedirs('results')
    if plt_save:
        if not os.path.exists('results/png'):
            os.makedirs('results/png')
        plt.savefig('results/png/' + filename + '.png')
