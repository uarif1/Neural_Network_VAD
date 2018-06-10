
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


def stdspectrum(s, m, f):
    '''
    STDSPECTRUM Generate standard acoustic/speech spectra in s- or z-domain
    [B,A,SI,SN]=(S,M,F,N,ZI,BS,AS)

    Parameters
    ----------

    s: type
      Spectrum type (either text or number - see below) or 0 to use bs/as
    m: type
      mode: char 1 specifies output type (ONLY power spectrum supported),
                p - power spectrum

    f: type
      set of frequencies in Hz (modes 'f','m','p','d')

    Returns
    -------
    type

        b   (1) numerator of the output spectrum (modes 's' or 'z')
            (2) output waveform (mode 't')
            (3) outptut spectrum (modes 'f', 'm', 'p' or 'd')
        a   (1) denonminator of the output spectrum (modes 's' or 'z')
            (2) final state of the filter - use as the zi input of a future
            call (mode 't')
        si  spectrum type number (0 to 10)
        sn  spectrum name

    Spectrum type
    13  LTASS-1994   : the long-term average speech spectrum that is taken
    from Table 2 in [2]

    Obsolete fits included for backward compatibility only:

    5  X1-LTASS-P50  : (use 11 instead) the long-term average speech spectrum
        taken from Table 1 of [4].
    6  X1-LTASS-1994 : (use 13 instead) the long-term average speech spectrum
        that is taken from Table 2 in [2]
    12  X2-LTASS-1994 : (use 13 instead) the long-term average speech spectrum
        that is taken from Table 2 in [2]
    References:
    [1]	Methods for the calculation of the speech intelligibility index.
    ANSI Standard S3.5-1997 (R2007), American National Standards Institute,
    1997.
    [2]	D. Byrne, H. Dillon, K. Tran, S. Arlinger, K. Wilbraham, R. Cox,
    B. Hayerman,
    R. Hetu, J. Kei, C. Lui, J. Kiessling, M. N. Kotby, N. H. A. Nasser,
    W. A. H. E. Kholy, Y. Nakanishi, H. Oyer, R. Powell, D. Stephens,
    R. Meredith,
    T. Sirimanna, G. Tavartkiladze, G. I. Frolenkov, S. Westerman, and
    C. Ludvigsen.
    An international comparison of long-term average speech spectra.
    JASA, 96 (4): 2108–2120, Oct. 1994.
    [3]	CENELEC. Electroacoustics - sound level meters. T
    echnical Report EN EN 61672-1:2003, 2003.
    (also ANSI S1.42-2001)
    [4]	ITU-T. Artificial voices. Standard P.50, Sept. 1999.
    [5]   ITU-T. Measurement of weighted noise in sound-programme circuits.
    Recommendation J.16, 1988.
    [6]   ITU-R. Measurement of audio-requency noise voltage level in sound
    broadcasting. Recommendation BS.468., 1986.
    [7]   NRSC AM Reemphasis, Deemphasize, and Broadcast Audio Transmission
    Bandwidth Specifications,
    EIA-549 Standard, Electronics Industries Association , July 1988.
    [8]   NRSC AM Reemphasis, Deemphasize, and Broadcast Audio Transmission
    Bandwidth Specifications,
    NRSC-1-A Standard, Sept 2007,
    Online: http://www.nrscstandards.org/SG/NRSC-1-A.pdf
    [9]   H. Fletcher and W. A. Munson. Loudness, its definition, measurement
    and calculation. J. Acoust Soc Amer, 5: 82–108, Oct. 1933.
    [10]  American National Standard Specification for Sound Level Meters.
    ANSI S1.4-1983 (R2006)/ANSI S1.4a-1985 (R2006), American National Standards
    Institute
    [11]	IEEE standard equipment requirements and measurement techniques for
    analog transmission
    parameters for telecommunications. Standard IEEE Std 743-1995, Dec. 1995.
    Other candidates: (a) Z-weighting, (b) ISO226, (c) P.48 spectra

    Copyright (C) Mike Brookes 2008
    Version: $Id: stdspectrum.m 8211 2016-07-20 20:59:16Z dmb $

    VOICEBOX is a MATLAB toolbox for speech processing.
    Home page: http://www.ee.ic.ac.uk/hp/staff/dmb/voicebox/voicebox.html
    '''

    # sn = 'X1-LTASS-1994'
    sb = np.array([1.97204932e+07, 8.91167289e+11, 1.09886653e+17,
                   2.25097646e+21, 0, 0, 0])

    sa = np.array([1, 1.71706427e+05, 8.44605379e+09, 7.29748112e+14,
                   2.25553520e+18, 6.24272042e+21, 3.46143239e+24,
                   2.53519335e+27])

    # sp = [-253.31327847 + 672.1085551j,
    #       -1299.18854371 + 2301.20640564j,
    #       -10646.95262798+68290.70281603j,
    #       -147307.51763333 + 0.j,
    #       -253.31327847 - 672.1085551j,
    #       -1299.18854371 - 2301.20640564j,
    #       -10646.95262798-68290.70281603j]
    #
    # sz = [0. + 0.j,      0. + 0.j,
    #       0. + 0.j, -22550.63757895 + 0.j,
    #       -11319.6356104 + 70239.17710766j, -11319.6356104 - 70239.17710766j]

    h = np.polyval(sb, 2j*np.pi*f)/np.polyval(sa, 2j*np.pi*f)
    b = h*np.conj(h)
    return b
