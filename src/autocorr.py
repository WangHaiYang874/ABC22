from random import sample
import numpy as np
from scipy import fft

def next_pow_2(n):
    ret = 1
    while ret < n:
        ret = ret << 1
    return ret

def acf_1d(X):
    n = len(X)
    N = next_pow_2(n)
    FX = fft.fft(X,2*N)
    acov = fft.ifft(FX * np.conjugate(FX))[:n].real
    return acov/acov[0]

def acf_nd(samples):
    m,n = samples.shape
    return np.array([acf_1d(samples[:,i]) for i in range(n)]).T