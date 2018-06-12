
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

from numpy.fft import fft
from numpy.matlib import repmat


def rdct(x, n=None, a=None, b=1):
    '''
    RDCT     Discrete cosine transform of real data Y=(X,N,A,B)
    Data is truncated/padded to length N.

    This routine is equivalent to multiplying by the matrix

    rdct(eye(n)) = diag([sqrt(2)*B/A repmat(2/A,1,n-1)]) * cos((0:n-1)'*(0.5:n)*pi/n)

    Default values of the scaling factors are A=sqrt(2N) and B=1 which
    results in an orthogonal matrix. Other common values are A=1 or N and/or B=1 or sqrt(2).
    If b~=1 then the columns are no longer orthogonal.

    '''

    m, k = x.shape
    if n is None:
        n = m
    if a is None:
        a = np.sqrt(n)
    if n > m:
        x = np.hstack((x, np.zeros(n-m, k)))
    elif n < m:
        x = np.vstack((x[:n+1, :], x[m:, :]))

    x = np.vstack((x[np.arange(0, n, 2), :],
                   x[np.arange(2*np.fix(n/2), 1, -2), :]))

    z = np.concatenate(([np.sqrt(2)],
                        2*np.exp((-0.5*1j*np.pi/n)*np.arange(1, n))))[np.newaxis].T

    y = np.real(fft(x)*repmat(z, 1, k))/a
    y[0, :] = y[1, :]*b
    return y
