'''
This program is free software
you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation
either version 2 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY
without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You can obtain a copy of the GNU General Public License from
http: // www.gnu.org/copyleft/gpl.html or by writing to
Free Software Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
'''

import numpy as np
from fromfrq import frq2bark


def erb2frq(erb):
    '''
    ERB2FRQ  Convert ERB frequency scale to Hertz FRQ=(ERB)
    frq = erb2frq(erb) converts a array of ERB-rate values
    to the corresponding frequencies in Hz.
    [frq,bnd] =  erb2frq(erb) also calculates the ERB bandwidths
    Note that erb values will be clipped to 43.032 which corresponds to
    infinite frequency.
    The inverse function is frq2erb.
    The erb scale is measured using the notched-noise method [3].
    We have df/de = 6.23*f^2 + 93.39*f + 28.52
    where the above expression gives the Equivalent Rectangular
    Bandwidth (ERB)in Hz  of a human auditory filter with a centre
    frequency of f kHz.
    By integrating the reciprocal of the above expression, we
    get:
        e = k ln((f/p-1)/(f/q-1))/d
    where p and q are the roots of the equation: -0.312 and -14.7
        and d = (6.23*(p-q))/1000 = 0.08950404
    from this we can derive:
    f = k/(h-exp(d*e)) + c
    where k = 1000 q (1 - q/p) = 676170.4
          h = q/p = 47.06538
          c = 1000q = -14678.49
    and f is in Hz
    Note that the maximum permissible value of e is log(b)/c=43.032 since this
    gives f=inf

    Parameters
    ----------
    erb : numpy.array
        array of ERB-rate values.

    Returns
    -------
    tuple
        frq : of frequencies in Hz.
        bnd :

    TODO: GRAPH, bnd

    References:
      [1] B.C.J.Moore & B.R.Glasberg "Suggested formula for
          calculating auditory-filter bandwidth and excitation
          patterns", J Acoust Soc America V74, pp 750-753, 1983
      [2] O. Ghitza, "Auditory Models & Human Performance in Tasks
          related to Speech Coding & Speech Recognition",
          IEEE Trans on Speech & Audio Processing, Vol 2,
          pp 115-132, Jan 1994
     [3] R. D. Patterson. Auditory filter shapes derived with noise
         stimuli. J. Acoust. Soc. Amer., 59: 640–654, 1976.

      Copyright (C) Mike Brookes 1998
      Version: $Id: erb2frq.m 5749 2015-03-01 16:01:14Z dmb $
      VOICEBOX is a MATLAB toolbox for speech processing.
      Home page: http://www.ee.ic.ac.uk/hp/staff/dmb/voicebox/voicebox.html
    '''

    u = np.array([6.23e-6, 93.39e-3, 28.52])
    p = np.sort(np.roots(u))  # p = np.array([-14678.5, - 311.9])
    d = 1e-6*(6.23*(p[1]-p[0]))  # d = 0.0895
    c = p[0]  # c = -14678.5
    k = p[0] - p[0] ** 2/p[1]  # k = 676170.4
    h = p[0]/p[1]  # h = 47.06538

    frq = np.multiply(np.sign(erb), k/np.maximum(h-np.exp(d*np.abs(erb)), 0)+c)
    bnd = np.polyval(u, abs(frq))
    return frq, bnd


def mel2frq(mel):
    '''
    MEL2FRQ  Convert Mel frequency scale to Hertz FRQ=(MEL)
    frq = mel2frq(mel) converts a array of Mel frequencies
    to the corresponding real frequencies.
    mr gives the corresponding gradients in Hz/mel.
    The Mel scale corresponds to the perceived pitch of a tone

    The relationship between mel and frq is given by [1]:

    m = ln(1 + f/700) * 1000 / ln(1+1000/700)

    This means that m(1000) = 1000

    Parameters
    ----------
    mel : numpy.array
        array of mel values.

    Returns
    -------
    tuple
        f : array of frequencies in Hz.
        mr : gradients in Hz/mel.

    TODO: GRAPH

    References:

        [1] J. Makhoul and L. Cosell. "Lpcw: An lpc vocoder with
            linear predictive spectral warping", In Proc IEEE Intl
            Conf Acoustics, Speech and Signal Processing, volume 1,
            pages 466–469, 1976. doi: 10.1109/ICASSP.1976.1170013.
        [2] S. S. Stevens & J. Volkman "The relation of pitch to
            frequency", American J of Psychology, V 53, p329 1940
        [3] C. G. M. Fant, "Acoustic description & classification
            of phonetic units", Ericsson Tchnics, No 1 1959
            (reprinted in "Speech Sounds & Features", MIT Press 1973)
        [4] S. B. Davis & P. Mermelstein, "Comparison of parametric
            representations for monosyllabic word recognition in
            continuously spoken sentences", IEEE ASSP, V 28,
            pp 357-366 Aug 1980
        [5] J. R. Deller Jr, J. G. Proakis, J. H. L. Hansen,
            "Discrete-Time Processing of Speech Signals", p380,
            Macmillan 1993

      Copyright (C) Mike Brookes 1998
      Version: $Id: mel2frq.m 1874 2012-05-25 15:41:53Z dmb $

      VOICEBOX is a MATLAB toolbox for speech processing.
      Home page: http://www.ee.ic.ac.uk/hp/staff/dmb/voicebox/voicebox.html

    '''

    k = 1000/np.log(1+1000/700)    # 1127.01048
    frq = np.multiply(700*np.sign(mel), np.exp(np.abs(mel)/k)-1)
    mr = (700+np.abs(frq))/k
    return frq, mr


def bark2frq(b, m=''):
    '''
    BARK2FRQ  Convert the BARK frequency scale to Hertz FRQ=(BARK)

    The Bark scale was defined by an ISO committee and published in [2]. It
    was based on a varienty of experiments on the thresholds for complex
    sounds, masking, perception of phase and the loudness of complex
    sounds. The Bark scale is named in honour of Barkhausen, the creator
    of the unit of loudness level [2]. Critical band k extends
    from bark2frq(k-1) to bark2frq(k). The inverse function is frq2bark.

    There are many published formulae approximating the Bark scale.
    The default is the one from [1] but with a correction at high and
    low frequencies to give a better fit to [2] with a continuous derivative
    and ensure that 0 Hz = 0 Bark.
    The h and l mode options apply the corrections from [1] which are
    not as good and do not give a continuous derivative. The H and L
    mode options suppress the correction entirely to give a simple formula.
    The 's' option uses the less accurate formulae from [3] which have been
    widely used in the lterature.

    Parameters
    ----------
    b : numpy.array
        matrix of frequencies in Bark.
    m : mode options
        'h'   use high frequency correction from [1]
        'l'   use low frequency correction from [1]
        'H'   do not apply any high frequency correction
        'L'   do not apply any low frequency correction
        'u'   unipolar version: do not force b to be an odd function
              This has no effect on the default function which is odd anyway
        's'   use the expression from Schroeder et al. (1979)
        'g'   plot a graph TODO

    Returns
    -------
    tuple
        f : frequency values in Hz
        c : Critical bandwidth: d(freq)/d(bark)

   [1] H. Traunmuller, Analytical Expressions for the
       Tonotopic Sensory Scale”, J. Acoust. Soc. Am. 88,
       1990, pp. 97-100.
   [2] E. Zwicker, Subdivision of the audible frequency range into
       critical bands, J Accoust Soc Am 33, 1961, p248.
   [3] M. R. Schroeder, B. S. Atal, and J. L. Hall. Optimizing digital
       speech coders by exploiting masking properties of the human ear.
       J. Acoust Soc Amer, 66 (6): 1647–1652, 1979. doi: 10.1121/1.383662.

      Copyright (C) Mike Brookes 2006-2010
      Version: $Id: bark2frq.m 4501 2014-04-24 06:28:21Z dmb $

   VOICEBOX is a MATLAB toolbox for speech processing.
   Home page: http://www.ee.ic.ac.uk/hp/staff/dmb/voicebox/voicebox.html
   '''
    A = 26.81
    B = 1960
    C = -0.53
    E = A+C
    D = A*B
    P = 0.53/(3.53) ** 2
    V = 3-0.5/P
    W = V ** 2 - 9
    Q = 0.25
    R = 20.4
    xy = 2
    S = 0.5*Q/xy
    T = R+0.5*xy
    U = T-xy
    X = T*(1+Q)-Q*R
    Y = U-0.5/S
    Z = Y**2 - U**2

    if 'u' in m:
        a = b
    else:
        a = np.abs(b)
    if 's' in m:
        f = 650*np.sinh(a/7)
    else:
        if 'l' in m:
            m1 = a < 2
            a[m1] = (a[m1]-0.3)/0.85
        elif 'L' not in m:
            m1 = a < 3
            a[m1] = V+np.sqrt(W+a[m1]/P)
        if 'h' in m:
            m1 = a > 20.1
            a[m1] = (a[m1]+4.422)/1.22
        elif 'H' not in m:
            m2 = a > X
            m1 = (a > U) & ~m2
            a[m2] = (a[m2]+Q*R)/(1+Q)
            a[m1] = Y+np.sqrt(Z+a[m1]/S)
        f = D*np.power(E-a, -1)-B
    if 'u' not in m:
        f = np.multiply(f, np.sign(b))  # force to be odd
    _, c = frq2bark(f, m)
    return f, c


def cent2frq(c):
    '''
    FRQ2ERB  Convert Hertz to Cents frequency scale [C,CR]=(FRQ)
    frq = frq2mel(c) converts a vector of frequencies in cents
    to the corresponding values in Hertz.
    100 cents corresponds to one semitone and 440Hz corresponds to 5700
    cents.
    The cr output gives the gradient in Hz/cent.

    The relationship between cents and frq is given by:

        c = 1200 * log2(f/(440*(2^((3/12)-5)))

    Parameters
    ----------
    c : np.array
        Cents freq Scale

    Returns
    -------
    tuple :
        frq : of frequencies in Hz
        cr : the gradient in Hz/cent.

    # TODO: plot graph

    Reference:

     [1] Ellis, A.
         On the Musical Scales of Various Nations
         Journal of the Society of Arts, 1885, 485-527
      Copyright (C) Mike Brookes 1998
      Version: $Id: cent2frq.m 3123 2013-06-19 19:03:53Z dmb $

    VOICEBOX is a MATLAB toolbox for speech processing.
    Home page: http://www.ee.ic.ac.uk/hp/staff/dmb/voicebox/voicebox.html
    '''
    p = 1200/np.log(2)
    q = 5700-p*np.log(440)
    # c = np.multiply(1200*np.sign(frq), np.log2(frq/(440*2 ** ((3.0/12)-5))))
    af = np.exp((np.abs(c)-q)/p)
    frq = np.multiply(np.sign(c), af)
    cr = af/p
    return frq, cr
