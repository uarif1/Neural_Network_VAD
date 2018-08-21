
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
    """
    Given a vector with shape = (no of features, time stamps) return the
    original feature vector along with the first and second derivatives
    with shape (3* no of features, time stamps)

    Parameters
    ----------
    features : np.array
        features with shape (no of features, time stamps)

    Returns
    -------
    np.array
        original features along with first and second derivatives in shape
        (3* no of features, time stamps)

    """
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


def Neural_Network_VAD(speech, fs=1, filename='VADres', show_plt=True,
                       plt_save=True, csvfile=True):
    """
    perform Speech/Non-speech identification at 25 ms frames giving output
    as 1 (speech) or 0 (Non-speech) and save the results in a png plot and csv
    file

    Parameters
    ----------
    speech : np.array
        mono speech signal: 1-D array
    fs : float
        sampling frequency.
    filename : str
        filename to be used for saving png and csv results.
    show_plt : bool
        option to show plot.
    plt_save : bool
        option to save plot.
    csvfile : bool
        option to save results in csv.

    Returns
    -------
    tuple:
        Speech/Non-Speech identification results using:
            Convolution LSTM Dense Neural Network
            LSTM Dense Neural Network

    """

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

    # model = load_model('Conv1D_LSTM_1_frame.h5')
    # pred_conv_lstm = model.predict(features.T[np.newaxis])[0, :, 1].T > 0.5

    model = load_model('Conv_LSTM.h5')
    pred_conv_lstm = model.predict(features.T[np.newaxis])[0].T[0] > 0.5

    # model = load_model('LSTM_1_frame.h5')
    # pred_lstm_only = model.predict(features.T[np.newaxis])[0, :, 1].T > 0.5

    model = load_model('LSTM.h5')
    pred_lstm_only = model.predict(features.T[np.newaxis])[0].T[0] > 0.5

    fig = plt.figure()
    fig.clf()
    plt.subplot(3, 1, 1)
    plt.plot(np.arange(len(speech)) / fs, speech, label='speech')
    plt.title('speech amplitude')
    plt.xlabel('time')
    plt.ylabel('amplitude')
    plt.subplot(3, 1, 2)
    plt.plot(tx, pred_conv_lstm, label='Convolution_LSTM_Dense_Neural_Network')
    plt.xlabel('time')
    plt.title('Convolution_LSTM_Dense_Neural_Network')
    plt.yticks([0, 1], ['non-speech', 'speech'])
    plt.subplot(3, 1, 3)
    plt.plot(tx, pred_lstm_only, label='LSTM_Dense Neural Network')
    plt.xlabel('(time (s)')
    plt.title('LSTM_Dense_Neural_Network')
    plt.yticks([0, 1], ['non-speech', 'speech'])
    fig = plt.gcf()
    fig.set_size_inches(12, 7, forward=True)

    # save plot of specified
    if plt_save or csvfile:
        if not os.path.exists('results'):
            os.makedirs('results')
    if plt_save:
        if not os.path.exists('results/png'):
            os.makedirs('results/png')
        plt.savefig('results/png/' + filename + '.png')

    # show plot if specicided
    if show_plt:
        plt.show()

    # save in csv file if specified
    if csvfile:
        if not os.path.exists('results/csv'):
            os.makedirs('results/csv')
        csvarr = np.vstack((tx, np.around(pred_conv_lstm), pred_lstm_only))
        np.savetxt('results/csv/' + filename + '.csv', csvarr.T, delimiter=',',
                   header='time,Convolution_LSTM prediction,LSTM_prediction',
                   comments='', fmt='%.4f, %1d, %1d')

    return pred_conv_lstm, pred_lstm_only
