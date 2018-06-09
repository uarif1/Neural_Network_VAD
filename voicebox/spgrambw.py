
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

from numpy.matlib import repmat
from numpy.fft import fft

from filtbankm import filtbankm


def spgrambw(s, fs, mode='', bw=200, frange=None, tinc=0):
    '''
    SPGRAMBW Calculate spectrogram [T,F,B]=(s,fs,mode,bw,fmax,db,tinc,ann)

    For examples of the many options available see:
    http://www.ee.ic.ac.uk/hp/staff/dmb/voicebox/tutorial/spgrambw/spgram_tut.pdf

    Parameters
    ----------
    s : np.array
        speech signal, or single-sided power spectrum array, S(NT,NF), in
        power per Hz
    fs : float
        sample fequency (Hz)
    bw : float
        bandwidth resolution in Hz (DFT window length = 1.81/BW)[default: 200]
    frange : np.array
        frequency range [Fmin Fstep Fmax]. If all 3 are omittend then Fstep is
        is taken to be (Fmax-Fmin)/257, Fmin is taken to be 0 and Fmax is taken
        to be FS/2.
    tinc : float
        output frame increment in seconds [0 or missing uses default=0.45/BW]

    Returns
    -------
    tuple:

        t : T(NT)= time axis values (in seconds). Input sample s(n) is at time
                   offset+n/fs.
        f : F(NF)= frequency axis values in Hz or, unless mode=H, other
                   selected frequency units according to mode: m=mel,
                   l=log10(Hz), b=bark,e=erb-rate
        b : B(NT,NF)= spectrogram values in power per x where x depends on the
                      'pPmbel' options clipped to DB range if 'D' option and
                      in dB if 'd' option.


        The BW input gives the 6dB bandwidth of the Hamming window used in the
        analysis. Equal amplitude frequency components are guaranteed to give
        separate peaks if they are this far apart. This value also determines
        the time resolution: the window length is 1.81/BW and the low-pass
        filter applied to amplitude modulations has a 6-dB bandwidth of
        BW/2 Hz.

        The units are power per Hz unless the 'P' option is given in which case
        power per displayed unit is used or power per decade for the 'l'
        option.

        Copyright (C) Mike Brookes 1997-2011
        Version: $Id: spgrambw.m 6803 2015-09-12 09:31:44Z dmb $

        VOICEBOX is a MATLAB toolbox for speech processing.
        Home page: http://www.ee.ic.ac.uk/hp/staff/dmb/voicebox/voicebox.html
   '''

    # first decode the input arguments
    if frange is None:
        frange = [0, fs/2, (fs/2)/257.0]

    firstframef = 1/fs  # first sample or frame is at time 1/fs fs(2)

    # flmin = 30  # min frequency for 'l' option
    nfrq = 257  # default number of frequency bins
    # fnyq = fs/2

    # fmin to fmax in steps of finc
    fx = np.arange(frange[0], frange[2]+1, frange[1])
    nfrq = len(fx)

    f = fx

    # now calculate the spectrogram

    winlen = np.fix(1.81*fs/bw)  # window length
    # Hamming window
    win = 0.54+0.46*np.cos(np.arange(1-winlen, winlen, 2)*np.pi/winlen)
    ninc = max(np.round(tinc*fs), 1)  # window increment in samples
    #  we need to take account of minimum freq increment + make it exact
    #  if possible
    # enough oversampling to get good interpolation
    fftlen = nextpow2(4*winlen)
    win = win/np.sqrt(sum(win ** 2))  # ensure window squared sums to unity

    sf, t = enframe(s, win, ninc)
    t = firstframef+(t-2)/fs
    # time axis
    b = rfft(sf, fftlen, 1)
    b = b*np.conj(b)*2/fs  # Power per Hz
    # correct for no negative zero frequency to double the power
    b[:, 0] = b[:, 0]*0.5
    # correct for no negative nyquist frequency to double the power
    b[:, -1] = b[:, -1]*0.5
    # fb = np.arange(fftlen/2)*fs/fftlen  # fft bin frequencies

    filtbankamp, cf = filtbankm(fftlen, fs, nfrq, fx[0], fx[-1], 'cush')
    b = np.matmul(b, filtbankamp.T)

    return t, f, b


def nextpow2(i):

    n = 1
    while n < i:
        n *= 2
    return n


def enframe(s, win, hop):
    '''
    %ENFRAME split signal up into (overlapping) frames: one per row.
    [F,T]=(X,WIN,HOP)

    Parameters
    ----------
    s : np.array
        speech signal
    win : np.array
        window
    hop : int
        frame increment in samples

    Returns
    -------
    tuple
        f :   enframed data - one frame per row
        t :   fractional time in samples at the centre of each frame
              with the first sample being 1.

    '''
    nx = len(s)
    lw = len(win)  # lw
    nli = nx-lw+hop
    nf = int(max(np.fix(nli/hop), 0))  # number of full frames
    f = np.zeros((nf, lw))
    indf = hop*np.arange(nf)
    inds = np.arange(lw).astype(int)

    f = s[repmat(indf, lw, 1).T.astype(int) + repmat(inds, nf, 1)]
    f = f*repmat(win, nf, 1)
    t0 = (1+lw)/2
    t = t0+hop*np.array(range(nf))
    return f, t


def rfft(x, n, d):
    '''
    RFFT     Calculate the DFT of real data Y=(X,N,D)
    Data is truncated/padded to length N if specified.
       N even:	(N+2)/2 points are returned with
                the first and last being real
       N odd:	(N+1)/2 points are returned with the
                first being real
    In all cases fix(1+N/2) points are returned
    D is the dimension along which to do the DFT
    Copyright (C) Mike Brookes 1998
    Version: $Id: rfft.m 713 2011-10-16 14:45:43Z dmb $

    VOICEBOX is a MATLAB toolbox for speech processing.
    Home page: http://www.ee.ic.ac.uk/hp/staff/dmb/voicebox/voicebox.html
    '''

    s = np.asarray(x.shape)
    if (s[0] * s[1]) == 1:
        return x
    else:
        y = fft(x, n, d)
        s[d] = 1+np.fix(n/2)
        return y[:, :s[d]]
