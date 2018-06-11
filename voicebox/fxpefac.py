
'''
   This program is free software; you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation; either version 2 of the License, or
   (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You can obtain a copy of the GNU General Public License from
   http://www.gnu.org/copyleft/gpl.html or by writing to
   Free Software Foundation, Inc.,675 Mass Ave, Cambridge, MA 02139, USA.
'''

import numpy as np
import scipy.ndimage

from numpy.matlib import repmat

from spgrambw import spgrambw
from filtbankm import filtbankm
from stdspectrum import stdspectrum
from v_findpeaks import v_findpeaks
from gaussmixp import gaussmixp


def fxpefac(s, fs, tinc=0.01, fstep=5, fmax=4000, fres=20, fbanklo=10,
            mpsmooth=21, maxtranf=1000, shortut=7, numopt=3, pefact=1.8,
            flim=[60, 400],  rampk=1.1, rampcz=100, tmf=2,
            w=[1.0000, 0.8250, 0.01868, 0.006773, 98.9, -0.4238],
            sopt='ilcwpf'):
    '''
    FXPEFAC PEFAC pitch tracker [FX,TT,PV,FV]=(S,FS,TINC,M,PP)

    Parameters
    ----------
    s : np.array
        speech signal
    fs : float
        Sample frequency (Hz)

    tinc : float
        Time increment between frames (s) [default: 0.01]

    fstep
        frequency resolution of initial spectrogram(Hz) =5
    fmax
        maximum frequency of initial spectrogram(Hz) =4000
    fres
        bandwidth of initial spectrogram(Hz) =20
    fbanklo
        low frequency limit of log filterbank(Hz) =10
    mpsmooth
        width of smoothing filter for mean power =21
    maxtranf
        maximum value of tranf cost term =1000
    shortut
        max utterance length to average power of entire utterance =7
    numopt
        number of possible frequencies per frame =3
    pefact
        shape factor in PEFAC filter =1.8
    flim
        range of feasible fundamental frequencies(Hz) =[60, 400]
    rampk
        constant for relative-amplitude cost term=1.1
    rampcz
        relative amplitude cost for missing peak=100
    tmf
        median frequency smoothing interval(s)=2
    w
        DP weights =[1.0000, 0.8250, 0.01868, 0.006773, 98.9, -0.4238]
    sopt
        spectrogram options='ilcwpf'


    Returns
    -------
    tuple:
        fx(nframe)     Estimated pitch (Hz)
        tx(nframe)     Time at the centre of each frame (seconds).
        pv(nframe)     Probability of the frame of being voiced
        fv             dictionary containing feature vectors
                       fv.vuvfea(nframe,2) = voiced/unvoiced GMM features

                            vuvfea: voiced-unvoiced features
                            best: best selected path
                            ff: pitch candidates
                            amp: pitch candidate amplitudes
                            medfx: median pitch
                            w: DP weights
                            dffact: df scale factor

 References
  [1]  S. Gonzalez and M. Brookes. PEFAC - a pitch estimation algorithm robust
       to high levels of noise.
       IEEE Trans. Audio, Speech, Language Processing, 22 (2):518-530,Feb.2014.
       doi: 10.1109/TASLP.2013.2295918.
  [2]  S.Gonzalez and M. Brookes,
       A pitch estimation filter robust to high levels of noise (PEFAC),
       Proc EUSIPCO,Aug 2011.
 Bugs/Suggestions
 (1) do long files in chunks
 (2) option of n-best DP
           Copyright (C) Sira Gonzalez and Mike Brookes 2011
      Version: $Id: fxpefac.m 10135 2017-09-27 07:15:56Z dmb $

   VOICEBOX is a MATLAB toolbox for speech processing.
   Home page: http://www.ee.ic.ac.uk/hp/staff/dmb/voicebox/voicebox.html
   '''

    # voiced/unvoiced decision based on 2-element feature vector
    # (a) mean power of the frame's log-freq spectrum (normalized so
    # its short-term average is LTASS)
    # (b) sum of the power in the first three peaks

    w_u = np.array([0.1461799, 0.3269458, 0.2632178,
                    0.02331986, 0.06360947, 0.1767271]).T

    m_u = np.array([[13.38533, 0.4199435],
                    [12.23505, 0.1496836],
                    [12.76646, 0.2581733],
                    [13.69822, 0.6893078],
                    [9.804372, 0.02786567],
                    [11.03848, 0.07711229]])

    v_u = np.array([[0.4575519, 0.002619074, 0.002619074, 0.01262138],
                    [0.7547719, 0.008568089, 0.008568089, 0.001933864],
                    [0.5770533, 0.003561592, 0.003561592, 0.00527957],
                    [0.3576287, 0.01388739, 0.01388739, 0.04742106],
                    [0.9049906, 0.01033191, 0.01033191, 0.0001887114, ],
                    [0.637969, 0.009936445, 0.009936445, 0.0007082946]]).T\
        .reshape(6, 2, 2)

    w_v = np.array([0.1391365, 0.221577, 0.2214025,
                    0.1375109, 0.1995124, 0.08086066])

    m_v = np.array([[15.36667, 0.8961554],
                    [13.52718, 0.4809653],
                    [13.95531, 0.8901121],
                    [14.56318, 0.6767258],
                    [14.59449, 1.190709],
                    [13.11096, 0.2861982]])

    v_v = np.array([[0.196497, -0.002605404, -0.002605404, 0.05495016],
                    [0.6054919, 0.007776652, 0.007776652, 0.01899244],
                    [0.5944617, 0.0485788, 0.0485788, 0.03511229],
                    [0.3871268, 0.0292966, 0.0292966, 0.02046839],
                    [0.3377683, 0.02839657, 0.02839657, 0.04756354],
                    [1.00439, 0.03595795, 0.03595795, 0.006737475]]).T\
        .reshape(6, 2, 2)

    # dpwtdef = [1.0000, 0.8250, 0.01868, 0.006773, 98.9, -0.4238]

    # Spectrogram of the mixture
    fmin = 0
    [tx, f, MIX] = spgrambw(s, fs, '', fres, [fmin, fstep, fmax], tinc)
    MIX = np.real(MIX)
    nframes = len(tx)
    txinc = tx[1]-tx[0]  # actual frame increment
    # we could combine spgrambw and filtbankm into a single call to spgrambw
    # or use fft directly
    # Log-frequency scale
    [trans, cf] = filtbankm(2*len(f)-1, 2*f[-1], len(f), fbanklo, f[-1],
                            'usl')
    Ospec = np.matmul(MIX, trans.T)
    # Original spectrum in Log-frequency scale

    ##################################################################
    # Amplitude Compression

    # Calculate alpha based on LTASS ratios
    # uses an old version of the LTASS spectrum but we will need to recalculate
    # the GMM if we update it
    ltass = stdspectrum(6, 'p', cf)
    auxf = np.concatenate(([cf[0]], (cf[:-1]+cf[1:])/2, [cf[-1]]))
    ltass = ltass*np.diff(auxf)  # weight by bin width

    # estimated ltass
    Ospec = np.multiply(Ospec, repmat(np.diff(auxf), nframes, 1))
    # weight spectrum by bin width
    # O1 = Ospec

    if tx[-1] < shortut:  # if it is a short utterance
        eltass = np.mean(Ospec, 0)  # mean power per each frequency band
        eltass = smooth(eltass, mpsmooth)  # smooth in log frequency
        eltass = eltass.T  # force a row vector

        # Linear AC
        alpha = ltass/eltass
        alpha = alpha.T
        alpha = repmat(alpha, nframes, 1)
        Ospec = np.real(np.multiply(Ospec, alpha))  # force O to have an average LTASS spectrum

        # == == should perhaps exclude the silent portions ** *
    else:  # long utterance

        tsmo = 3  # time smoothing over 3 sec
        stt = np.round(tsmo/txinc)
        eltass = timesm(Ospec, stt)
        eltass = smooth(eltass, mpsmooth)  # filter in time and log frequency

        # Linear AC
        alpha = repmat(ltass, nframes, 1)/(eltass)
        Ospec = np.real(np.multiply(Ospec, alpha))

    ##########################################################################
    # Create the filter to detect the harmonics
    ini = np.where(cf > 3*cf[0])[0]
    # bin frequencies start at approximately 0.33 with sca(ini(1)) = 1 exactly
    sca = cf/cf[ini[0]]

    # Middle
    # restrict to 0.5 - 10.5 times fundamental
    sca = sca[np.logical_and(sca < 10.5, sca > 0.5)]

    sca1 = sca
    filh = 1/(pefact-np.cos(2*np.pi*sca1))
    tempvar = np.diff(np.concatenate(([sca1[0]], (sca1[:-1]+sca1[1:])/2,
                                      [sca1[-1]])))
    filh = filh - np.sum(filh*tempvar)/np.sum(tempvar)

    # == == this should just equal ini(1) == ==
    posit = np.where(sca >= 1)[0]
    negat = np.where(sca < 1)[0]
    numz = int(len(posit)-1-len(negat))
    filh = filh/np.max(filh)
    # length is always odd with central value = 1
    filh = np.concatenate((np.zeros((1, numz))[0], filh))

    ###################################################################
    # Filter the log-frequency scaled spectrogram
    # does a convolution with zero lag at centre of filh

    B = scipy.ndimage.correlate(Ospec, filh[np.newaxis], mode='constant')

    # flim = permitted fx range = [60 400]
    pfreq = np.where(np.logical_and(cf > flim[0], cf < flim[1]))[0]
    ff = np.zeros((nframes, numopt))
    amp = np.zeros((nframes, numopt))
    for i in range(nframes):
        # min separation = 5Hz @ fx = flim[0](could pre-calculate) == ==
        [pos, peak] = v_findpeaks(B[i, pfreq], 5/(cf[pfreq[1]]-cf[pfreq[0]]))
        if len(pos) > 0:
            ind = np.argsort(peak, kind='mergesort')[::-1]
            peak = np.sort(peak, kind='mergesort')[::-1]
            pos = pos[ind]  # indices of peaks in the B array
            # frequencies of peaks
            posff = cf[pfreq[pos]]
            fin = min(numopt, len(posff))
            ff[i, : fin] = posff[: fin]
            # save both frequency and amplitudes
            amp[i, : fin] = peak[: fin]

    #######################################################################
    # Probabilitly of the frame of being voiced

    # voiced/unvoiced decision based on 2-element feature vector
    # (a) mean power of the frame's log-freq spectrum(normalized so its
    # short-term average is LTASS)
    # (b) sum of the power in the first three peaks

    pow = np.asarray(np.mean(Ospec, 1)).T[0]

    vuvfea = np.hstack((np.log(pow)[np.newaxis].T,
                        (1e-3*np.sum(amp, 1)/(pow+1.75*1e5))[np.newaxis].T))

    ##################################################

    # Log probability of being unvoiced
    pru, _, _, _ = gaussmixp(vuvfea, m_u, v_u, w_u)
    # Log probability of being voiced
    prv, _, _, _ = gaussmixp(vuvfea, m_v, v_v, w_v)

    pv = np.transpose((1+np.exp(pru-prv)) ** -1)[0]

    #######################################################################
    # Dynamic programming

    # w[0]: relative amp, voiced local cost
    # w[1]: median pitch deviation cost
    # w[2]: df cost weight
    # w[3]: max df cost
    # w[4]: relative amp cost for missing peaks(very high)
    # w[5]: df mean

    # Relative amplitude
    camp = -amp/repmat(np.max(amp, 1)[np.newaxis].T, 1, numopt)
    # relative amplitude used as cost
    camp[amp == 0] = w[4]  # If no frequency found

    # Time interval for the median frequency
    inmf = int(np.round(tmf/txinc))

    # -----------------------------------------------------------------
    # FORWARDS
    # Initialize values
    cost = np.zeros((nframes, numopt))
    prev = np.zeros((nframes, numopt))
    medfx = np.zeros((nframes, 1))
    dffact = 2/txinc

    # First time frame
    # cost(1, : ) = w(1)*ramp(1, : )
    cost[0, :] = w[0]*camp[0, :]
    # only one cost term for first frame
    minfposidx = int(min(inmf, len(pv)-1))
    fpos = ff[:minfposidx, 0]
    # calculate median frequency of first 2 seconds

    mf = np.median(fpos[pv[0:minfposidx] > 0.6])
    if np.isnan(mf):
        mf = np.median(fpos[pv[: minfposidx] > 0.5])
        if np.isnan(mf):
            mf = np.median(fpos[pv[: minfposidx] > 0.4])
            if np.isnan(mf):
                # == == clumsy way of ensuring that we take the best frames ==
                mf = np.median(fpos[pv[: minfposidx] > 0.3])
                if np.isnan(mf):
                    mf = 0
    medfx[0] = mf

    for i in range(1, nframes):  # main dynamic programming loop
        if i > inmf-1:
            fpos = ff[i-inmf: i, 0]
            # fpos is the highest peak in each frame
            # find median frequency over past 2 seconds
            mf = np.median(fpos[pv[: inmf] > 0.6])
            if np.isnan(mf):
                mf = np.median(fpos[pv[: inmf] > 0.5])
                if np.isnan(mf):
                    mf = np.median(fpos[pv[: inmf] > 0.4])
                    if np.isnan(mf):
                        #  clumsy way of ensuring that we take the best frames
                        mf = np.median(fpos[pv[: inmf] > 0.3])
                        if np.isnan(mf):
                            mf = 0
        medfx[i] = mf
        # Frequency difference between candidates and cost
        df = dffact*(repmat(ff[i, :][np.newaxis].T, 1, numopt)
                     - repmat(ff[i-1, :][np.newaxis], numopt, 1))\
            / (repmat(ff[i, :][np.newaxis].T, 1, numopt)
               + repmat(ff[i-1, :][np.newaxis], numopt, 1))

        costdf = w[2]*np.minimum((df-w[5]) ** 2, w[3])

        # Cost related to the median pitch
        if mf == 0:  # this test was inverted in the original version
            costf = np.zeros((1, numopt))
        else:
            costf = np.abs(ff[i, :] - mf)/mf
        # == == should we allow the possibility of skipping frames ?
        prev[i, :] = np.argmin(costdf + repmat(cost[i-1, :], numopt, 1), 1)
        cost[i, :] = np.min(costdf + repmat(cost[i-1, :], numopt, 1), 1)
        # add on costs that are independent of previous path
        cost[i, :] = cost[i, :]+w[1]*costf + w[0]*camp[i, :]

    # Traceback

    fx = np.zeros((nframes, 1))
    ax = np.zeros((nframes, 1))
    best = np.zeros((nframes, 1)).astype(int)

    nose = np.where(cost[-1, :] == np.min(cost[-1, :]))[0]
    # == == bad method(dangerous)
    best[-1] = nose[0]
    fx[-1] = ff[-1, best[-1]]
    ax[-1] = amp[-1, best[-1]]
    for i in range(nframes-1, 0, -1):
        best[i-1] = prev[i, best[i]]
        fx[i-1] = ff[i-1, best[i-1]]
        ax[i-1] = amp[i-1, best[i-1]]

    fv = {'vufea': vuvfea, 'best': best, 'ff': ff, 'amp': amp, 'medfx': medfx,
          'w': w, 'dffact': dffact,
          'hist': np.hstack((np.asarray(np.log(np.mean(Ospec, 1))),
                             (np.sum(amp, 1)
                              / np.asarray(np.mean(Ospec, 1)).T[0])[np.newaxis].T))}

    return fx, tx, pv, fv


def smooth(x, n):
    # snx = x.shape[1]
    nf = x.shape[0]
    c = np.cumsum(x, 1)
    y = np.hstack((
        c[:, np.arange(0, n+1, 2)]/repmat(np.arange(1, n+1, 2), nf, 1),
        (c[:, n:]-c[:, :-n])/n,
        (repmat(c[:, -1], 1, int(np.floor(n/2)))-c[:, np.arange(-1-n+2, -1, 2)])
        / repmat(np.arange(n-2, 0, -2), nf, 1)))
    return y


def timesm(x, n):

    if not np.mod(n, 2):
        n += 1
    nx = x.shape[1]
    # nf = x.shape[0]
    c = np.cumsum(x, 0)
    n = int(n)
    mid = int(np.round(n/2))
    y = np.vstack((c[mid:n, :]/repmat(np.arange(mid+1, n+1)[np.newaxis].T, 1, nx),
                   (c[n:, :]-c[: -n, :])/n,
                   (repmat(c[-1, :], mid, 1) - c[-1-n+1: -1-mid, :])
                   / repmat(np.arange(n-1, mid, -1)[np.newaxis].T, 1, nx)))

    return y
