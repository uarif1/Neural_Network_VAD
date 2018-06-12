import numpy as np

from numpy.fft import fft


def rfft(x, n=None, d=None):
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
    if d is None:
        d = np.where(s > 1)[0][0]
    if n is None:
        n = s[d]

    if (s[0] * s[1]) == 1:
        return x
    else:
        y = fft(x, n, d)
        s[d] = 1+np.fix(n/2)
        return y[:s[0], :s[1]]
