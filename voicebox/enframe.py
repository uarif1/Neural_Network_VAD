import numpy as np

from numpy.matlib import repmat


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
