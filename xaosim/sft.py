''' ===================================================================
Test dodumentation for SFT, the "slow" Fourier Transform using the 
theory described in: http://adsabs.harvard.edu/abs/2007OExpr..1515935S

Example:
-------

>> import pupil
>> import sft
>> a = pupil.uniform_disk((200,200), 100)
>> ca_focal = sft.sft(a, 50, 20)

Will compute the complex amplitude in the focal plane for a pupil
over a +/- 10 lambda/D field of view

>> lyot = sft.isft(ca_focal, 200, 20)

Will compute the complex amplitude in a pupil plane located downstream 
from the previous focal plane, using only the spatial frequencies 
present in the array that was previously computed.
'''

import numpy as np

# ===================================================================
def sft(A2, NB, m, inv=False):
    ''' --------------------------------------------------------------
    Explicit Fourier Transform, using the theory described in:
    http://adsabs.harvard.edu/abs/2007OExpr..1515935S

    Assumes the original array is square.
    No need to "center" the data on the origin.

    Parameters:
    ----------

    - A2 : the 2D original array
    - NB : the linear size of the result array (integer)
    - m  : m/2 = maximum spatial frequency to be computed (in l/D)
    - inv: boolean (direct or inverse) see the definition of isft()
    -------------------------------------------------------------- '''

    NA    = np.shape(A2)[0]
    m     = float(m)
    coeff = m/(NA*NB)
    
    U = np.zeros((1,NB))
    X = np.zeros((1,NA))
    
    X[0,:] = (1./NA)*(np.arange(NA)-NA/2.)
    U[0,:] =  (m/NB)*(np.arange(NB)-NB/2.)
    
    sign = -1.0
    if inv:
        sign = 1.0
        
    A1 = np.exp(sign * 2j*np.pi* np.dot(np.transpose(U),X))
    A3 = np.exp(sign * 2j*np.pi* np.dot(np.transpose(X),U))

    B  = np.dot(np.dot(A1,A2),A3)

    return coeff*np.array(B)



# ===================================================================
def isft(A2, NB, m):
    ''' --------------------------------------------------------------
    Explicit inverse Fourier Transform, using the theory described in:
    http://adsabs.harvard.edu/abs/2007OExpr..1515935S

    See documentation for sft().
    -------------------------------------------------------------- '''
    return sft(A2, NB, m, inv=True)
