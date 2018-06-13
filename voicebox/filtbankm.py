
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

import collections
import re

import numpy as np

from numpy.matlib import repmat
from scipy.sparse import csr_matrix

from voicebox.fromfrq import frq2mel, frq2erb, frq2bark
from voicebox.tofrq import mel2frq, erb2frq, bark2frq


def filtbankm(n, fs, p=None, fl=None, fh=None, w='f'):
    '''
    FILTBANKM determine matrix for a linear/mel/erb/bark-spaced filterbank

    NOTE: THIS FUNCTION HAS ONLY BEEN TESTED ON THE 'usl' OPTION
    NOTE: THIS FUNCTION DOES NOT RETURN THE LOWEST AND HIGHEST NON-ZERO FFT BIN

    Usage: # TODO: correct examples

    Parameters
    ----------
    n : int or list
        length of fft or [nf df fa] nfrq=number of input frequency bins,
        frequency increment (Hz), first bin freq (Hz).

    fs : float
        sample rate in Hz.

    p : int
        number of filters in filterbank or the filter spacing in k-mel/bark/erb
        [default ceil(4.6*log10(2*(fa+(nf-1)*df)))].

    fl : float
        low end of the lowest filter in Hz (see 'h' option)
        [default = 0 or set to 30Hz for 'l' option].

    fh : float
         high end of highest filter in Hz [default = fs/2].

    w : str
        any sensible combination of the following (default is 'f', use only one
        of 'l', 'e', 'b', 'm' in the combination):
            'b' = bark scale instead of mel
            'e' = erb-rate scale
            'l' = log10 Hz frequency scale
            'f' = linear frequency scale [default]
            'm' = mel frequency scale
            'n' - round to the nearest FFT bin so each row of x contains only
                  one non-zero entry
            'c' = fl & fh specify centre of low and high filters instead of
                  edges
            'h' = fl & fh are in mel/erb/bark/log10 instead of Hz
            'H' = cf outputs are in mel/erb/bark/log10 instead of Hz
            'y' = lowest filter remains at 1 down to 0 frequency and
                  highest filter remains at 1 up to nyquist freqency
                  The total power in the fft is preserved (unless 'u' is
                  specified).
            'Y' = extend only at low frequency end (or high end if 'y' also
                  specified)
            'p' = input P specifies the number of filters [default if P>=1]
            'P' = input P specifies the filter spacing [default if P<1]
            'u' = input and output are power per Hz instead of power.
            'U' = input is power but output is power per Hz.
            's' = single-sided input: do not include symmetric negative
                  frequencies (i.e. non-DC inputs have been doubled)
            'S' = single-sided output: do not mirror the non-DC filter
                  characteristics (i.e. double non-DC outputs)

    Returns
    -------
    tuple:

        x : a sparse matrix containing the filterbank amplitudes
            If the il and ih outputs are given then size(x)=[p,ih-il+1]
            otherwise size(x)=[p,1+floor(n/2)]
            Note that the peak filter values equal 2 to account for the power
            in the negative FFT frequencies.

        cf :  the filterbank centre frequencies in Hz (see 'H' option)

     The routine performs interpolation of the input spectrum by convolving
     the power spectrum with a triangular filter and then simulates a
     filterbank with asymetric triangular filters.

     References:

     [1] S. S. Stevens, J. Volkman, and E. B. Newman. A scale for the measurement
         of the psychological magnitude of pitch. J. Acoust Soc Amer, 8: 185–19, 1937.
     [2] S. Davis and P. Mermelstein. Comparison of parametric representations for
         monosyllabic word recognition in continuously spoken sentences.
         IEEE Trans Acoustics Speech and Signal Processing, 28 (4): 357–366, Aug. 1980.
     Bugs/Suggestions
     (1) default frequencies won't work if the h option is specified
     (2) low default frequency is invalid if the 'l' option is specified
          Copyright (C) Mike Brookes 1997-2009

    VOICEBOX is a MATLAB toolbox for speech processing.

    '''
    # Note "FFT bin_0" assumes DC = bin 0 whereas "FFT bin_1" means DC = bin 1

    # wr = ' '  # default warping is linear frequency
    # for i = 1:length(w)
    #     if any(w(i) == 'lebm')
    #         wr = w(i)
    #     end
    # end
    # TODO: above for loop probably not needed
    if len(re.findall('l|e|b|m', w)) > 1:
        raise ValueError("use only one of 'l', 'e', 'b', 'm' for w")
    if len(re.findall('p|P', w)) > 1:
        raise ValueError("use only one of 'p' or 'P' for w")
    repeat_dict = collections.defaultdict(lambda: 0)
    for val in w:
        repeat_dict[val] += 1
        if repeat_dict[val] > 1:
            raise ValueError('repeated options in w')

    if fh is None:
        fh = 0.5*fs  # max freq is the nyquist
    if fl is None:
        fl = 30 if 'l' in w else 0
        # min freq is 30 Hz for log scale else its DC

    fa = 0
    if isinstance(n, (collections.Sequence, np.ndarray)):
        nf = n[0]  # number of input frequency bins
        df = n[1]  # input frequency bin spacing
        if len(n) > 2:
            fa = n[2]  # frequency of first bin
    else:
        nf = int(1+np.floor(n/2))  # number of input frequency bins
        df = fs/n  # input frequency bin spacing
    fin0 = fa + np.arange(nf) * df  # input frequency bins

    mflh = [fl, fh]
    if 'h' not in w:  # convert Hz to mel/erb/...
        if 'm' in w:
            mflh = frq2mel(mflh)  # convert frequency limits into mel
        if 'l' in w:
            if fl <= 0:
                raise ValueError('Low frequency limit must be >0 for l option')
            mflh = np.log10(mflh)  # convert frequency limits into log10 Hz
        if 'e' in w:
            mflh = frq2erb(mflh)  # convert frequency limits into erb-rate
        if 'b' in w:
            mflh = frq2bark(mflh)  # convert frequency limits into bark

    melrng = mflh[1] - mflh[0]  # mel/erb/... range
    # fn2 = np.floor(n/2) # bin index of highest positive frequency
    # (Nyquist if n is even)
    if p is None:
        # default number of output filters
        p = np.ceil(4.6*np.log10(2*(fa+(nf-1)*df)))

    if 'p' in w or (p < 1 and 'p' not in w):
        p = np.round(melrng/(p*1000))+1

    if 'c' in w:  # c option: specify fiter centres not edges
        melinc = melrng/(p-1)
        mflh = mflh+np.array([-1, 1])*melinc
    else:
        melinc = melrng/(p+1)

    # --------- Calculate the FFT bins0 corresponding to the filters-------

    # centre frequencies in mel/erb/... including dummy ends
    cf = mflh[0]+np.arange(p+2)*melinc
    cf[1:] = np.maximum(cf[1:], 0)  # only the first point can be negative
    # convert centre frequencies from mel/erb/... to Hz
    if 'l' in w:
        mb = np.power(10, (cf))
    elif 'e' in w:
        mb = erb2frq(cf)
    elif 'b' in w:
        mb = bark2frq(cf)
    elif 'm' in w:
        mb = mel2frq(cf)
    else:
        mb = cf

    # first sort out 2-sided input frequencies
    #
    fin = fin0
    fin = np.append(fin, fin[-1]+df)  # add on a dummy point at the high end
    if fin[0] == 0:
        fin = np.concatenate((-fin[nf:0:-1], fin))
    elif fin[0] <= df/2:
        fin = np.concatenate((-fin[nf::-1], fin))
    elif fin[0] < df:
        fin = np.concatenate((-fin[nf::-1], [fin[0]-df], [df-fin[0]], fin))
    elif fin[0] == df:
        fin = np.concatenate((-fin[nf::-1], [0], fin))
    else:
        fin = np.concatenate((-fin[nf::-1], [df-fin[0]], [fin(0)-df], fin))

    nfin = len(fin)  # length of extended input frequency list

    # now sort out the interleaving
    #
    fout = mb  # output frequencies in Hz
    lowex = len(re.findall('y|Y', w)) == 2  # extend to 0 Hz
    highex = 'y' in w and (fout[-2] < fin[-1])  # extend at high end
    if lowex:
        fout = np.concatenate(([0, 0], fout[1:]))
    if highex:
        fout = np.concatenate((fout[:-1], [fin[-1], fin[-1]]))

    mfout = len(fout)
    if 'u' in w or 'U' in w:
        gout = np.subtract(fout[2:], fout[: -2])
        gout = 2*np.power((gout+(gout == 0)), -1)  # Gain of output triangles
    else:
        gout = np.ones((mfout-2))

    if 'S' in w:
        msk = fout[1: 1] != 0
        # double non-DC outputs for a 1-sided output spectrum
        gout[msk] = 2*gout[msk]
    if 'u' in w:
        gin = np.ones((nfin-2))
    else:
        gin = np.subtract(fin[2:], fin[: -2])
        gin = np.power(2*(gin+(gin == 0)), -1)  # Gain of input triangles

    msk = fin[1: -1] == 0
    if 's' in w:
        # halve non-DC inputs to change back to a 2-sided spectrum
        gin[np.logical_not(msk)] = 0.5*gin[np.logical_not(msk)]
    if lowex:
        # double DC input to preserve its power
        gin[msk] = 2*gin[msk]

    foutin = np.concatenate((fout, fin))
    nfall = len(foutin)
    # left width
    wleft = np.concatenate(([0], fout[1:]-fout[:-1], [0], fin[1:]-fin[:-1]))

    wright = np.append(wleft[1:], 0)  # right width
    # gain of triangle posts
    ffact = np.concatenate(([0], gout, [0, 0], gin[:min(nf, nfin-nf-2)],
                            np.zeros((max(nfin-2*nf-2, 0))),
                            gin[nfin-nf-2: nfin-2], [0]))
    # ffact(wleft+wright == 0) = 0 # disable null width triangles shouldn't
    # need this if all frequencies are distinct
    # fall = np.sort(foutin)
    ifall = np.argsort(foutin, kind='mergesort')
    jfall = np.zeros(nfall, dtype=int)
    infall = np.arange(nfall)
    jfall[ifall] = infall  # unsort->sort index
    # zap nodes that are much too small/big
    ffact[ifall[np.concatenate((np.arange(0, max(jfall[0], jfall[mfout])-1,
                                          dtype=np.int32),
                                np.arange(min(jfall[mfout-1], jfall[nfall-1])
                                          + 2, nfall, dtype=np.int32)))]] = 0

    nxto = np.cumsum(ifall <= mfout-1)
    nxti = np.cumsum(ifall > mfout-1)
    # next input node to the right of each value(or nfall if none)
    nxtr = np.minimum(nxti+1+mfout, nfall)
    # next post to the right of opposite type(unsorted indexes)
    nxtr[ifall > mfout-1] = 1+nxto[ifall > mfout-1]
    # next post to the right of opposite type(converted to unsorted indices)
    # or if none: nfall/(mfout+1)
    nxtr = nxtr[jfall]

    # each triangle is "attached" to the node at its extreme right end
    # the general result for integrating the product of two trapesiums with
    # heights(a, b) and (c, d) over a width x is (ad+bc+2bd+2ac)*w/6
    #
    # integrate product of lower triangles

    msk0 = ffact > 0
    # select appropriate triangle pairs(unsorted indices)
    msk = msk0 & (ffact[nxtr-1] > 0)
    # unsorted indices of leftmost post of pair
    ix1 = infall[msk]  # dont subtract anything while using ix1 as index
    # unsorted indices of rightmost post of pair
    jx1 = nxtr[msk]-1
    # length of right triangle to the left of the left post
    vfgx = foutin[ix1]-foutin[jx1-1]
    yx = np.minimum(wleft[ix1], vfgx)
    # integration length
    wx1 = ffact[ix1]*ffact[jx1]*yx \
        * (wleft[ix1]*vfgx-yx*(0.5*(wleft[ix1]+vfgx)-yx/3)) \
        / (wleft[ix1]*wleft[jx1]+(yx == 0))

    # integrate product of upper triangles

    nxtu = np.maximum(np.append(nxtr[1:]-1, 0), 1)
    msk = msk0 & (ffact[nxtu-1] > 0)
    ix2 = infall[msk]  # unsorted indices of leftmost post of pair
    jx2 = nxtu[msk]-1  # unsorted indices of rightmost post of pair
    # length of left triangle to the right of the right post
    vfgx = foutin[ix2+1]-foutin[jx2]
    yx = np.minimum(wright[ix2], vfgx)  # integration length
    yx[foutin[jx2+1] < foutin[ix2+1]] = 0  # zap invalid triangles
    wx2 = ffact[ix2]*ffact[jx2]*(yx**2)*((0.5*(wright[jx2]-vfgx)+yx/3))\
        / (wright[ix2]*wright[jx2-1]+(yx == 0))

    # integrate lower triangle and upper triangle that ends to its right

    nxtu = np.maximum(nxtr-1, 1)
    msk = msk0 & (ffact[nxtu-1] > 0)
    ix3 = infall[msk]  # unsorted indices of leftmost post of pair
    jx3 = nxtu[msk] - 1  # unsorted indices of rightmost post of pair
    # length of upper triangle to the left of the lower post
    vfgx = foutin[ix3]-foutin[jx3]
    yx = np.minimum(wleft[ix3], vfgx)  # integration length
    yx[foutin[jx3+1] < foutin[ix3]] = 0  # zap invalid triangles
    wx3 = ffact[ix3]*ffact[jx3]*yx *\
        (wleft[ix3]*(wright[jx3]-vfgx)+yx
         * (0.5*(wleft[ix3]-wright[jx3]+vfgx)-yx/3))\
        / (wleft[ix3]*wright[jx3]+(yx == 0))

    # integrate upper triangle and lower triangle that starts to its right

    nxtu = np.concatenate((nxtr[1:], [1]))
    msk = msk0 & (ffact[nxtu-1] > 0)
    ix4 = infall[msk]  # unsorted indices of leftmost post of pair
    jx4 = nxtu[msk]-1  # unsorted indices of rightmost post of pair
    # length of upper triangle to the left of the lower post
    vfgx = foutin[ix4+1]-foutin[jx4-1]
    yx = np.minimum(wright[ix4], vfgx)
    # integration length
    wx4 = ffact[ix4]*ffact[jx4]*yx**2*(0.5*vfgx-yx/3)\
        / (wright[ix4]*wleft[jx4]+(yx == 0))

    # now create the matrix

    iox = np.sort(np.vstack((
        np.concatenate((ix1, ix2, ix3, ix4)),
        np.concatenate((jx1, jx2, jx3, jx4))
    )), axis=0)
    # [iox; [wx1 wx2 wx3 wx4] > 0]
    msk = iox[1, :]+1 <= (nfall+mfout)/2
    # convert negative frequencies to positive
    iox[1, msk] = (nfall+mfout-1)-iox[1, msk]
    if highex:
        iox[0, iox[0, :] == mfout-1] = mfout-2  # merge highest two output node
    if lowex:
        iox[0, iox[0, :] == 2] = 3  # merge lowest two output nodes
    x = csr_matrix((np.concatenate((wx1, wx2, wx3, wx4)),
                    (iox[0, :]-1-lowex, np.maximum(iox[1, :]-nfall+nf+1, 0))),
                   shape=(p, nf))
    # TODO: no need to do sparse i think
    # x = np.concatenate((wx1, wx2, wx3, wx4))
    #
    # sort out the output argument options
    #
    if 'H' not in w:
        cf = mb  # output Hz instead of mel/erb/...
    cf = cf[1: p+1]  # remove dummy end frequencies
    if 'n' in w:  # round outputs to the centre of gravity bin
        sx2 = np.sum(x, 1)
        msk = sx2 != 0
        vxc = np.zeros((p, 1))
        vxc[msk] = np.round((x[msk, :]*np.arange(1, nf+1).T)/sx2[msk])
        x = csr_matrix(sx2, (np.arange(p), vxc), shape=(p, nf))

    # TODO: il and ih
    # il = 1
    # ih = nf

    # msk = np.any(x > 0, 1)
    # ilidx = np.nonzero(msk)[0]
    # if len(ilidx == 0):
    #     ih = 1
    #     il = None
    # elif len(ilidx == 1):
    #     il = 1
    #     ih = ilidx[0]
    # else:
    #     il = ilidx[0]
    #     ih = ilidx[1]
    # x = x[:, np.arange(il, ih)]

    if 'u' in w:
        sx = np.sum(x, 1)
        x = x/repmat(sx+(sx == 0), 1, x.shape[1])

    # TODO: plot graph
    return x, cf
