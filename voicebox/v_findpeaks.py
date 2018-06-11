
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


def v_findpeaks(y, w, x=None):
    '''
    V_FINDPEAKS finds peaks with optional quadratic interpolation
    [K,V]=(Y,M,W,X)


    Parameters
    ----------
    y : np.array
        COLUMN VECTOR Y(N,1) is the input signal
        (does not work with UInt datatype)
    w : type
        W  is the width tolerance; a peak will be eliminated if there is
        a higher peak within +-W. Units are samples or X values
    x : np.array
        X(N,1)   COLUMN VECTOR x-axis locations of Y values
        [default: 1:length(Y)]

    Returns
    -------
    tuple
        K(P,1)   are the positions in X of the peaks in Y (fractional if M='q')
        V(P,1)   are the peak amplitudes: if M='q' the amplitudes will be
                 interpolated whereas if M~='q' then V=Y(K).
    Outputs are column vectors regardless of whether Y is row or column.
    If there is a plateau rather than a sharp peak, the routine will place the
    peak in the centre of the plateau. When the W input argument is specified,
    the routine will eliminate the lower of any pair of peaks whose separation
    is <=W; if the peaks have exactly the same height, the second one will be
    eliminated.
    Unless the 'f' or 'l' options are given, all peak locations satisfy 1<K<N.


    Copyright (C) Mike Brookes 2005
    Version: $Id: v_findpeaks.m 6564 2015-08-16 16:56:40Z dmb $

    VOICEBOX is a MATLAB toolbox for speech processing.
    Home page: http://www.ee.ic.ac.uk/hp/staff/dmb/voicebox/voicebox.html
    '''

    ny = len(y)

    dx = y[1:]-y[: -1]
    r = np.where(dx > 0)[0]
    f = np.where(dx < 0)[0]
    k = []  # set defaults
    v = []
    if len(r) > 0 and len(f) > 0:  # we must have at least one rise & one fall

        dr = np.where(dx > 0)[0]+1
        dr[1:] = r[1:]-r[:-1]  # only 0 index diff from matlab
        rc = np.ones(ny)
        rc[r] = 1-dr
        rc[0] = 0
        rs = np.cumsum(rc)  # = time since the last rise

        df = np.where(dx < 0)[0]+1
        df[1:] = f[1:]-f[: -1]
        fc = np.ones(ny)
        fc[f] = 1-df
        fc[0] = 0
        fs = np.cumsum(fc)  # = time since the last fall

        rp = repmat(-1, ny, 1).reshape(ny)
        rp[np.concatenate(([0], r))] = np.concatenate((dr-1, [ny-r[-1]-1]))
        rq = np.cumsum(rp)  # = time to the next rise

        fp = repmat(-1, ny, 1).reshape(ny)
        fp[np.concatenate(([0], f))] = np.concatenate((df-1, [ny-f[-1]-1]))
        fq = np.cumsum(fp)  # = time to the next fall
        # the final term centres peaks within a plateau

        k = np.where((rs < fs) & (fq < rq) & (np.floor((fq-rs)/2) == 0))
        k = k[0] + 1
        if k[-1] == len(y):
            k = k[:-1]
        v = y[k]
        if x is not None:  # convert to the x-axis using linear interpolation
            k = x[k]
    if ny > 1:
        if w > 0:
            j = np.where(k[1:]-k[: -1] <= w)[0]
            while len(j) > 0:
                j = j+(v[j] >= v[j+1])
                k = np.delete(k, j)
                v = np.delete(v, j)
                j = np.where(k[1:]-k[: -1] <= w)[0]

    return k, v
