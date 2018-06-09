
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
from numpy.linalg import eig


def gaussmixp(y, m, v=None, w=None, a=None, b=None):
    '''
    GAUSSMIXP calculate probability densities from or plot a Gaussian mixture
    model

    Usage: (1) gaussmixp([],m,v,w)

    Parameters
    ----------
    n data values, k mixtures, p parameters, q data vector size

    y : type
        Y(n,q) = input data (or optional plot range if no out arguments)
        Row of Y(i,:) represents a single observation of the
        transformed GMM data point X: Y(i,1:q)=X(i,1:p)*A'+B'. If A and B are
        omitted and q=p, then Y(i,:)=X(i,:).
    m : np.array
        M(k,p) = mixture means for x(p)
    v : np.array
        V(k,p) or V(p,p,k) variances (diagonal or full)
    w : type
        W(k,1) = weights (default = np.ones(m.shape[0], m.shape[1])
    a : type
        A(q,p) = transformation: y=x*A'+ B' (where y and x are row vectors)
    b : type
        B(q,1)   If A is omitted or null, y=x*I(B,:)' where I is the identity
            matrix. If B is also omitted or null, y=x*I(1:q,:)'.
            Note that most commonly, q=p and A and B are omitted entirely.

    Returns
    -------
    tuple:

        lp : LP(n,1) = log probability of each data point
        rp : RP(n,k) = relative probability of each mixture
        kh : KH(n,1) = highest probability mixture
        kp : KP(n,1) = relative probability of highest probability mixture

    Copyright (C) Mike Brookes 2000-2009
    Version: $Id: gaussmixp.m 7339 2016-01-06 18:05:30Z dmb $

    VOICEBOX is a MATLAB toolbox for speech processing.
    Home page: http://www.ee.ic.ac.uk/hp/staff/dmb/voicebox/voicebox.html
    '''

    k, p = m.shape
    n, q = y.shape

    # TODO: Handle output if q is 0

    if v is None:
        v = np.ones(m.shape[0], m.shape[1])

    if w is None:
        w = repmat(1/k, k, 1)

    fv = v.ndim > 2 or v.shape[0] > k  # full covariance matrix is supplied
    if a is not None:
        m = m*a  # no offset b specified
        if b is not None:
            m = m*a+repmat(b, k, 1)  # offset b is specified
        v1 = v  # save the original covariance matrix array
        v = np.zeros((q, q, k))  # create new full covariance matrix array
        if fv:
            for ik in range(k):
                v[:, :, ik] = a*v1[:, :, ik]*a
        else:
            for ik in range(k):
                v[:, :, ik] = (a*repmat(v1[ik, :], q, 1))*a
            fv = 1  # now we definitely have a full covariance matrix
    elif q < p or b is not None:  # need to select coefficient subset
        if b is None:
            b = np.arange(q)
        m = m[:, b]
        if fv:
            v = v[b, b, :]
        else:
            v = v[:, b]

    memsize = 50e6  # set memory size to use

    lp = np.zeros((n, 1))
    rp = np.zeros((n, k))
    wk = np.ones((k, 1))
    if n > 0:
        if not fv:  # diagonal covariance
            ##############################
            # Diagonal Covariance matrices

            # If data size is large then do calculations in chunks

            # chunk size for testing data points
            nb = min(n, max(1, np.floor(memsize/(8*q*k))))
            nl = np.ceil(n/nb)  # number of chunks
            jx0 = n-(nl-1)*nb  # size of first chunk
            im = repmat(np.arange(k).T, nb, 1)
            wnb = np.ones((1, nb))
            wnj = np.ones((1, jx0))
            vi = -0.5*v ** (-1)  # data-independent scale factor in exponent
            # log of external scale factor(excluding - 0.5*q*log(2pi) term)
            lvm = np.log(w)-0.5*sum(np.log(v), 2)

            # first do partial chunk

            jx = jx0
            ii = np.arange(jx)
            kk = repmat(ii, k, 1)
            km = repmat(np.arange(k), 1, jx)
            py = np.sum(y[kk.flatten(), :] -
                        np.power(m[km.flatten(), :], 2*vi[km.flatten(), :]), 2)\
                .reshape((k, jx)) + lvm[:, wnj]
            # find normalizing factor for each data point to prevent
            # underflow when using exp()
            mx = np.max(py, 0)
            # find normalized probability of each mixture for each datapoint
            px = np.exp(py-mx[wk, :])
            # total normalized likelihood of each data point
            ps = np.sum(px, 0)
            # relative mixture probabilities for each data point
            # (columns sum to 1)
            rp[ii, :] = np.transpose(px/ps[wk, :])
            lp[ii] = np.log(ps)+mx

            for il in range(2, nl+1):
                ix = jx+1
                jx = jx+nb  # increment upper limit
                ii = np.arange(ix-1, jx)
                kk = repmat(ii, k, 1)
                py = sum((y[kk.flatten(), :]-np.power(m[im, :]),
                          2*vi[im, :]), 2).reshape(k, nb)+lvm[:, wnb]
                # find normalizing factor for each data point to prevent
                # underflow when using exp()
                mx = np.max(py, 1)
                # find normalized probability of each mixture for each
                # datapoint
                px = np.exp(py-mx[wk, :])
                # total normalized likelihood of each data point
                ps = np.sum(px, 1)
                # relative mixture probabilities for each data point
                # (columns sum to 1)
                rp[ii, :] = np.transpose(px/ps[wk, :])
                lp[ii] = np.log(ps)+mx
        else:
            ##########################
            # Full Covariance matrices
            pl = q*(q+1)/2
            lix = np.arange(q ** 2)
            cix = repmat(np.arange(q), q, 1)
            rix = cix.T  # index of lower triangular elements
            lix = lix[np.logical_not((cix < rix).flatten(1))]
            lixi = np.zeros((q, q))
            # lixi[lix] = np.arange(1, pl)
            np.put(lixi, lix, np.arange(0, pl), mode='wrap')
            lixi = np.transpose(lixi)
            # reverse index to build full matrices
            np.put(lixi, lix, np.arange(0, pl), mode='wrap')
            lixi = lixi.flatten().astype(int)
            # lixi[lix] = np.arange(1, pl+1)
            vt = v.reshape(q ** 2, k)
            vt = vt[lix, :].T  # lower triangular in rows

            # If data size is large then do calculations in chunks

            # chunk size for testing data points
            nb = min(n, max(1, np.floor(memsize/(24*q*k))))
            nl = np.ceil(n/nb).astype(int)  # number of chunks
            jx0 = int(n-(nl-1)*nb)
            # size of first chunk
            wnb = np.ones((1, nb))
            wnj = np.ones((1, jx0)).astype(int)

            vi = np.zeros((q*k, q))
            # stack of k inverse cov matrices each size q*q
            vim = np.zeros((q*k, 1))
            # stack of k vectors of the form inv(vt)*m
            mtk = np.zeros((q*k, 1))
            # stack of k vectors of the form m
            lvm = np.zeros((k, 1))
            wpk = np.array(list(range(q))*k).T

            for ik in range(k):

                # these lines added for debugging only
                # vk = vt[k, lixi].reshape( q, q)
                # condk[ik] = cond[vk]

                # convert lower triangular to full and find eigenvalues
                [dvk, uvk] = eig(vt[ik, lixi].reshape(q, q))
                if np.any(dvk <= 0):
                    raise Exception('Covariance matrix for mixture % d is not'
                                    + 'positive definite' % (ik))
                # calculate inverse
                vik = -0.5*np.matmul(np.matmul(uvk, np.diag(dvk ** (-1))),
                                     uvk.T)
                # vi contains all mixture inverses stacked on top of each other
                vi[ik*q+np.arange(q), :] = vik
                # vim contains vi*m for all mixtures stacked on top of each
                # other
                vim[ik*q+np.arange(q)] = np.matmul(vik, m[ik, :])[np.newaxis].T
                # mtk contains all mixture means stacked on top of each other
                mtk[ik*q+np.arange(q)] = np.array([m[ik, :]]).T
                # vm contains the weighted sqrt of det(vi) for each mixture
                lvm[ik] = np.log(w[ik])-0.5*np.sum(np.log(dvk))
            ########################
            # first do partial chunk

            jx = jx0
            ii = np.arange(jx)
            xii = y[ii, :].T

            py = np.sum(np.multiply((np.matmul(vi, xii)-repmat(vim, 1, jx0)),
                                    (xii[wpk, :]-repmat(mtk, 1, jx0)))
                        .reshape(q, jx*k, order='F'), 0)\
                .reshape(k, jx, order='F') + repmat(lvm, 1, jx0)
            # find normalizing factor for each data point to prevent underflow
            # when using exp()
            mx = np.max(py, 0)
            # find normalized probability of each mixture for each datapoint
            px = np.exp(py-repmat(mx, k, 1))  # px = np.exp(py-mx[wk, :])

            # total normalized likelihood of each data point
            ps = np.sum(px, 0)
            # relative mixture probabilities for each data point
            # (columns sum to 1)
            rp[ii, :] = np.transpose(px/repmat(ps, k, 1))
            lp[ii] = (np.log(ps)+mx)[np.newaxis].T

            for il in range(2, nl):
                ix = jx+1
                jx = jx+nb  # increment upper limit
                ii = np.arange(ix-1, jx)
                xii = y[ii, :].T
                py = np.sum(((np.matmul(vi, xii)-repmat(vim, 1, nb))
                             * (xii[wpk, :]-repmat(mtk, 1, nb))
                             ).reshape(q, nb*k, order='F'), 0).\
                    reshape(k, nb, order='F')+repmat(lvm, 1, nb)

                # find normalizing factor for each data point to prevent
                #  underflow when using exp()
                mx = np.max(py,  1)
                px = np.exp(py-repmat(mx, wk, 1))
                # find normalized probability of each mixture for each
                # datapoint
                ps = np.sum(px, 1)
                # total normalized likelihood of each data point
                rp[ii, :] = np.transpose(px/repmat(ps, wk, 1))
                # relative mixture probabilities for each data point
                # (columns sum to 1)
                lp[ii] = np.log[ps]+mx
        lp = lp-0.5*q*np.log(2*np.pi)
    else:
        raise Exception('incorrect dimension of y')
    kp = np.max(rp, 1)
    kh = np.argmax(rp, 1)

    # TODO: plot graph
    return lp, rp, kh, kp
