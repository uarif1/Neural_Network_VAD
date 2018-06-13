
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

from voicebox.enframe import enframe
from voicebox.rfft import rfft
from voicebox.melbankm import melbankm
from voicebox.rdct import rdct


def melcepst(s, fs=16000, nc=12, p=None, n=None, inc=None, fl=0, fh=0.5):
    '''
    MELCEPST Calculate the mel cepstrum of a signal C=(S,FS,W,NC,P,N,INC,FL,FH)

    Parameters
    ----------

        s	: np.array
            speech signal

        fs : float
            sample rate in Hz (default 16000)

        nc : int
            number of cepstral coefficients excluding 0'th coefficient
            [default 12]

        p  : int
            number of filters in filterbank
            [default: floor(3*log(fs)) =  approx 2.1 per ocatave]

        n  : int
            length of frame in samples [default power of 2 < (0.03*fs)]

        inc: float
            frame increment [default n/2]

        fl : float
            low end of the lowest filter as a fraction of fs [default = 0]

        fh : float
            high end of highest filter as a fraction of fs [default = 0.5]

        Hamming window in time domain

        triangular shaped filters in mel domain

        filters act in the absolute magnitude domain

        highest and lowest filters taper down to zero

    Returns
    -------

    tuple :

           c : mel cepstrum output: one frame per row.

           tc: fractional time in samples at the centre of each frame
               with the first sample being 1.

   '''

    if p is None:
        p = np.floor(3*np.log(fs))
    if n is None:
        n = nextpow2(0.03*fs)
    if inc is None:
        inc = np.floor(n/2)

    # Hamming window
    [z, tc] = enframe(s, 0.54-0.46*np.cos(2*np.pi
                                          * np.arange(0, n) / (n-1)), inc)
    f = rfft(z.T)
    m, _, a, b = melbankm(p, n, fs, fl, fh)
    pw = f[np.arange(a, b+1), :]*np.conj(f[np.arange(a, b+1), :])
    pth = np.max(pw)*1E-20
    ath = np.sqrt(pth)
    y = np.log(np.maximum(m * np.abs(f[np.arange(a, b+1), :]), ath))
    c = rdct(y).T
    nf = c.shape[0]
    nc = nc+1
    if p > nc:
        c = c[:, :nc]
    elif p < nc:
        c = np.concatenate((c, np.zeros((nf, nc-p))))
    c = c[:, 1:]  # get rid of 0th coeff
    # nc = nc-1

    return c, tc


def nextpow2(i):

    n = 1
    while n < i:
        n *= 2
    return n
