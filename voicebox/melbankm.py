
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

from scipy.sparse import csr_matrix

from voicebox.fromfrq import frq2mel
from voicebox.tofrq import mel2frq


def melbankm(p, n, fs, fl=0, fh=0.5):
    '''
    MELBANKM determine matrix for a mel/erb/bark-spaced filterbank
    [X,MN,MX]=(P,N,FS,FL,FH,W)
    This function uses triangular shaped filters in mel/erb/bark domain
    and has highest and lowest filters taper down to zero

    Parameters
    ----------

    p : int
        number of filters in filterbank or the filter spacing in k-mel/bark/erb

    n : int
        length of fft

    fs: int
        sample rate in Hz

    fl: float
        low end of the lowest filter as a fraction of fs [default = 0]

    fh: float
        high end of highest filter as a fraction of fs [default = 0.5]

    Returns
    -------
    x     a sparse matrix containing the filterbank amplitudes
          If the mn and mx outputs are given then size(x)=[p,mx-mn+1]
          otherwise size(x)=[p,1+floor(n/2)]
          Note that the peak filter values equal 2 to account for the power
          in the negative FFT frequencies.

    mc    the filterbank centre frequencies in mel/erb/bark

    mn    the lowest fft bin with a non-zero coefficient

    mx    the highest fft bin with a non-zero coefficient
          Note: you must specify both or neither of mn and mx.

     References:

     [1] S. S. Stevens, J. Volkman, and E. B. Newman. A scale for the measurement
         of the psychological magnitude of pitch. J. Acoust Soc Amer, 8: 185–19, 1937.
     [2] S. Davis and P. Mermelstein. Comparison of parametric representations for
         monosyllabic word recognition in continuously spoken sentences.
         IEEE Trans Acoustics Speech and Signal Processing, 28 (4): 357–366, Aug. 1980.

  '''

    # Note "FFT bin_0" assumes DC = bin 0 whereas "FFT bin_1" means DC = bin 1

    mflh = np.array([fl, fh])*fs

    mflh, _ = frq2mel(mflh)  # convert frequency limits into mel

    melrng = mflh[1] - mflh[0]  # mel range
    # bin index of highest positive frequency(Nyquist if n is even)
    fn2 = np.floor(n/2)

    if p < 1:
        p = np.round(melrng/(p*1000))-1
    melinc = melrng/(p+1)

    #
    # Calculate the FFT bins corresponding to
    # [filter  # 1-low filter#1-mid filter#p-mid filter#p-high]
    #
    blim = mel2frq(mflh[0]+np.array([0, 1, p, p+1])*melinc)[0]*n/fs

    mc = mflh[0]+np.arange(1, p+1)*melinc  # mel centre frequencies
    # lowest FFT bin_0 required might be negative)
    b1 = int(np.floor(blim[0])+1)
    b4 = int(min(fn2, np.ceil(blim[3])-1))
    # highest FFT bin_0 required
    #
    # now map all the useful FFT bins_0 to filter1 centres
    #
    pf = (frq2mel(np.arange(b1, b4+1)*fs/n)[0]-mflh[0])/melinc
    #
    # remove any incorrect entries in pf due to rounding errors
    #
    if pf[0] < 0:
        pf = pf[1:]
        b1 = b1+1
    if pf[-1] >= p+1:
        pf = pf[:-1]
        b4 = b4-1
    # FFT bin_0 i contributes to filters_1 fp(1+i-b1)+[0 1]
    fp = np.floor(pf)
    pm = pf-fp  # multiplier for upper filter
    k4 = len(fp)
    # FFT bin_1 k4+b1 is the last to contribute to any filters
    # FFT bin_1 k2+b1 is the first to contribute to both upper and lower
    # filters
    k2 = np.where(fp > 0)[0]
    k2 = k2[0] if len(k2) > 0 else k4+1
    # FFT bin_1 k3+b1 is the last to contribute to both upper and lower
    # filters
    k3 = np.where(fp < p)[0]
    k3 = k3[-1] if len(k3) > 0 else 0

    r = np.concatenate((1+fp[:k3+1], fp[k2: k4])
                       ).astype(int)  # filter number_1
    # FFT bin_1 - b1
    c = np.concatenate((np.arange(1, k3+2), np.arange(k2+1, k4+1)))
    v = np.concatenate((pm[:k3+1], 1-pm[k2:k4+1]))
    mn = b1  # lowest fft bin_1
    mx = b4  # highest fft bin_1
    if b1 < 0:
        # convert negative frequencies into positive
        c = np.abs(c+b1-1)-b1+1

    # double all except the DC and Nyquist(if any) terms
    # there is no Nyquist term if n is odd
    msk = (c+mn > 1) & (c+mn < n-fn2+1)
    v[msk] = 2*v[msk]
    #
    # sort out the output argument options
    #

    x = csr_matrix((v, (r-1, c-1)))
    return x, mc, mn, mx
