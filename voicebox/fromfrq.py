
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


def frq2bark(f, m=''):
    """
    FRQ2BARK  Convert Hertz to BARK frequency scale BARK=(FRQ)
           bark = frq2bark(frq) converts a array of frequencies (in Hz)
           to the corresponding values on the BARK scale.

    Parameters
    ----------
    f : numpy.array
        array of frequencies in Hz.
    m : str
        mode options (the default is '').
            'h'   use high frequency correction from [1]
            'l'   use low frequency correction from [1]
            'H'   do not apply any high frequency correction
            'L'   do not apply any low frequency correction
            'z'   use the expressions from Zwicker et al. (1980) for b and c
            's'   use the expression from Schroeder et al. (1979)
            'u'   unipolar version: do not force b to be an odd function
                 This has no effect on the default function which is odd anyway
            'g'   plot a graph:TODO

    Returns
    -------
    tuple
        b : bark values
        c : Critical bandwidth: d(freq)/d(bark)


       The Bark scale was defined by in ISO532 and published in [2]. It
       was based on a varienty of experiments on the thresholds for complex
       sounds, masking, perception of phase and the loudness of complex
       sounds. The Bark scale is named in honour of Barkhausen, the creator
       of the unit of loudness level[2]. Frequency f lies in critical
       band ceil(frq2bark(f)). The inverse function is bark2frq.


       There are many published formulae approximating the Bark scale.
       The default is the one from [1] but with a correction at high and
       low frequencies to give a better fit to[2] with a continuous derivative
       and ensure that 0 Hz = 0 Bark.
       The h and l mode options apply the corrections from [1] which are
       not as good and do not give a continuous derivative. The H and L
       mode options suppress the correction entirely to give a simple formula.
       The 's' option uses the less accurate formulae from [3] which have been
       widely used in the lterature.
       The 'z' option uses the formulae from [4] in which the c output
       is not exactly the reciprocal of the derivative of the bark function.


       [1] H. Traunmuller, Analytical Expressions for the
           Tonotopic Sensory Scale”, J. Acoust. Soc. Am. 88,
           1990, pp. 97-100.
       [2] E. Zwicker, Subdivision of the audible frequency range into
           critical bands, J Accoust Soc Am 33, 1961, p248.
       [3] M. R. Schroeder, B. S. Atal, and J. L. Hall. Optimizing digital
           speech coders by exploiting masking properties of the human ear.
           J. Acoust Soc Amer, 66 (6): 1647–1652, 1979. doi: 10.1121/1.383662.
       [4] E. Zwicker and E. Terhardt.  Analytical expressions for
           critical-band rate and critical bandwidth as a function of frequency
           J. Acoust Soc Amer, 68 (5): 1523–1525, Nov. 1980.


       The following code reproduces the graphs 3(c) and 3(d) from [1].
           b0 = (0: 0.5: 24)';
           f0 = [[2 5 10 15 20 25 30 35 40 45 51 57 63 70 77 ...
               84 92 100 108 117 127 137 148 160 172 185 200 ...
               215 232 250 270 290 315]*10 [34 37 40 44 48 53 ...
               58 64 70 77 85 95 105 120 135 155]*100]';
           b1 = frq2bark(f0);      b2 = frq2bark(f0, 'lh');
           b3 = frq2bark(f0, 'LH'); b4 = frq2bark(f0, 'z');
           plot(b0, [b0 b1 b2 b3 b4]-repmat(b0, 1, 5));
           xlabel('Frequency (Bark)'); ylabel('Error (Bark)');
           legend('Exact', 'voicebox', 'Traunmuller1990', ...
                  'Traunmuller1983', 'Zwicker1980', 'Location', 'South');

          Copyright (C) Mike Brookes 2006-2010
          Version: $Id: frq2bark.m 4501 2014-04-24 06:28:21Z dmb $

    """

    A = 26.81
    B = 1960
    C = -0.53
    D = A*B
    P = 0.53/(3.53) ** 2
    Q = 0.25
    R = 20.4
    xy = 2
    S = 0.5*Q/xy
    T = R+0.5*xy
    U = T-xy

    g = f if 'u' in m else np.absolute(f)

    if 'z' in m:
        b = 13*np.arctan(0.00076*g)+3.5*np.arctan((f/7500) ** 2)
        c = 25+75*(1+1.4e-6*f ** 2) ** 0.69
    elif 's' in m:
        b = 7*np.log(g/650+np.sqrt(1+(g/650)**2))
        c = np.cosh(b/7)*650/7
    else:
        b = A*g/(B+g)+C
        d = np.power(D*(B+g), (-2))
        if 'l' in m:
            m1 = b < 2
            d[m1] = d[m1]*0.85
            b[m1] = 0.3+0.85*b[m1]
        elif 'L' not in m:
            m1 = b < 3
            b[m1] = b[m1]+P*(3-b[m1]) ** 2
            d[m1] = np.multiply(d[m1], (1-2*P*(3-b[m1])))

        if 'h' in m:
            m1 = b > 20.1
            d[m1] = d[m1]*1.22
            b[m1] = 1.22*b[m1]-4.422
        elif 'H' not in m:
            m2 = b > T
            m1 = (b > U) & ~m2
            b[m1] = b[m1]+S*(b[m1]-U) ** 2
            b[m2] = (1+Q)*b[m2]-Q*R
            d[m2] = d[m2]*(1+Q)
            d[m1] = np.multiply(d[m1], (1+2*S*(b[m1]-U)))
        c = d ** (-1)
    if 'u' not in m:
        b = np.power(b, np.sign(f))

    return b, c


def frq2cent(frq):
    '''
    FRQ2ERB  Convert Hertz to Cents frequency scale [C,CR]=(FRQ)
    [c,cr] = frq2mel(frq) converts a array of frequencies (in Hz)
    to the corresponding values on the logarithmic cents scale.
    100 cents corresponds to one semitone and 440Hz corresponds to 5700
    cents.
    The optional cr output gives the gradient in Hz/cent.

    The relationship between cents and frq is given by:

    c = 1200 * log2(f/(440*(2^((3/12)-5)))

    Parameters
    ----------
    frq : numpy.array
        array of frequencies in Hz

    Return
    -------
    tuple
        c : Cents freq Scale
        cr : gradient in Hz/cent

    TODO: PLOT GRAPH

    Reference:

     [1] Ellis, A.
         On the Musical Scales of Various Nations
         Journal of the Society of Arts, 1885, 485-527
      Copyright (C) Mike Brookes 1998
      Version: $Id: frq2cent.m 3122 2013-06-19 19:02:47Z dmb $

    VOICEBOX is a MATLAB toolbox for speech processing.
    Home page: http://www.ee.ic.ac.uk/hp/staff/dmb/voicebox/voicebox.html
    '''
    p = 1200/np.log(2)
    q = 5700-p*np.log(440)
    af = np.abs(frq)
    c = np.multiply(1200*np.sign(frq), np.log2(frq/(440*2 ** ((3/12)-5))))
    c = np.multiply(np.sign(frq), (p*np.log(af)+q))
    cr = af/p
    return c, cr


def frq2erb(frq):
    '''
    FRQ2ERB  Convert Hertz to ERB frequency scale ERB=(FRQ)
    erb = frq2erb(frq) converts a array of frequencies (in Hz)
    to the corresponding values on the ERB-rate scale on which
    the human ear has roughly constant resolution as judged by
    psychophysical measurements of the cochlear filters. The
    inverse function is erb2frq.
    The erb scale is measured using the notched-noise method [3].

    We have df/de = 6.23*f^2 + 93.39*f + 28.52
    where the above expression gives the Equivalent Rectangular
    Bandwidth (ERB)in Hz  of a human auditory filter with a centre
    frequency of f kHz.

    By integrating the reciprocal of the above expression, we
    get:
            e = a ln((f/p-1)/(f/q-1))

    where p and q are the roots of the equation: -0.312 and -14.7
    and a = 1000/(6.23*(p-q)) = 11.17268

    We actually implement e as

            e = a ln (h - k/(f+c))

    where k = 1000(q - q^2/p) = 676170.42
     h = q/p = 47.065
          c = -1000q = 14678.49
    and f is in Hz

    Parameters
    ----------
    frq : numpy.array
        array of frequencies in Hz

    Returns
    -------
    tuple
        erb : erb frequency scale
        bnd :

    TODO: MAKE GRAPH, FINDOUT bnd desc

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
      Copyright (C) Mike Brookes 1998-2015
      Version: $Id: frq2erb.m 5749 2015-03-01 16:01:14Z dmb $

   VOICEBOX is a MATLAB toolbox for speech processing.
   Home page: http://www.ee.ic.ac.uk/hp/staff/dmb/voicebox/voicebox.html
   '''

    u = np.array([6.23e-6, 93.39e-3, 28.52])
    p = np.sort(np.roots(u))
    # p = [-14678.5 - 311.9]
    a = 1e6/(6.23*(p[1]-p[0]))
    # a = 11.17
    c = p[0]
    # c = -14678.5
    k = p[0] - p[0] ** 2/p[1]
    # k = 676170.42
    h = p[0]/p[1]
    # h = 47.065
    g = np.abs(frq)
    # erb= np.multiply(11.17268*np.sign(frq),np.log(1+46.06538*g/(g+14678.49)))
    erb = np.multiply(a*np.sign(frq), np.log(h-k/(g-c)))
    bnd = np.polyval(u, g)
    return erb, bnd


def frq2mel(frq):
    '''
    FRQ2ERB  Convert Hertz to Mel frequency scale MEL = (FRQ)
    [mel, mr] = frq2mel(frq) converts a array of frequencies(in Hz)
    to the corresponding values on the Mel scale which corresponds
    to the perceived pitch of a tone.
    mr gives the corresponding gradients in Hz/mel.
    The relationship between mel and frq is given by[1]:

    m = ln(1 + f/700) * 1000 / ln(1+1000/700)

    This means that m(1000) = 1000

    Parameters
    ----------
    frq : numpy.array
        array of frequencies in Hz.

    Returns
    -------
    tuple
        mel : Mel Scale array
        m : gradient in Hz/mel

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
    Copyright(C) Mike Brookes 1998
    Version: $Id: frq2mel.m 1874 2012-05-25 15: 41: 53Z dmb $

    VOICEBOX is a MATLAB toolbox for speech processing.
    Home page: http: // www.ee.ic.ac.uk/hp/staff/dmb/voicebox/voicebox.html

     '''

    k = 1000/np.log(1+1000/700)  # 1127.01048

    af = np.abs(frq)
    mel = np.multiply(np.sign(frq), np.log(1+af/700)*k)
    mr = (700+af)/k
    return mel, mr


def frq2midi(f):
    '''
    FRQ2MIDI Convert frequencies to musical note numbers [N,T]=(F)
    notes are numbered in semitones with middle C being 60
    Note 69 (the A above middle C) has a frequency of 440 Hz.
    These note numbers are used by MIDI. Note numbers are not necessarily
    integers.
    t is a text representation of the note in which
    C4# denotes C sharp in octave 4. Octave 4 goes
    from middle C up to the B above middle C. For the white
    notes on the piano, the third character is a space.
    Negative frequencies are equivalent to positive frequencies
    except that flats will be used instead of sharps. Thus
    C4# would become D4-
    see MIDI2FRQ for the inverse transform

    Parameters
    ----------
    f : np.array
        array of frequencies in Hz.

    Returns
    -------
    tuple
        n : musical note numbers
        t : text representation of the note


    Copyright (C) Mike Brookes 1998
    Version: $Id: frq2midi.m 713 2011-10-16 14:45:43Z dmb $
    VOICEBOX is a MATLAB toolbox for speech processing.
    Home page: http://www.ee.ic.ac.uk/hp/staff/dmb/voicebox/voicebox.html
    '''

    n = 69+12*np.log(np.abs(f)/440)/np.log(2)
    m = np.round(n)
    o = np.floor(m/12.0)-1
    m = np.int32(m-12*o+6*np.sign(f)-5)-1
    a = np.array(list('CDDEEFGGAABBCCDDEFFGGAAB'))
    b = np.array(list(' - -  - - -  # #  # # # '))
    t = list(map(''.join, zip(a[m], np.int32(np.mod(o, 10)).astype(str),
                              b[m])))
    return n, t
