import numpy as np


class Prior:
    '''
    Object of this class is essentially a probability distribution, 
    that is used as prior in the bayersian computation. 

    Attributes:
    - dim:dimention of theta
    - theta_range=None
        a dim*2 array that specifies the range of the theta. 
        default is None, and in this case it is considered to be 
        (-1,1) ** dim
    - pdf(theta) -> p(theta)
    - sampler() -> theta
        returns a random sample of theta
    '''

    def __init__(self, log_pdf, sampler, theta_range=None) -> None:

        self.log_pdf = log_pdf
        self.sampler = sampler
        self.dim = len(self.sampler())
        if theta_range is None:
            self.theta_range = [(-1, 1) for _ in range(self.dim)]
        else:
            assert(theta_range.shape == (2, self.dim))
            self.theta_range = theta_range

    def proba(self, theta):
        return np.exp(self.log_pdf(theta))
