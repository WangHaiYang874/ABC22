import Prior
import numpy as np
from matplotlib import pyplot as plt
from joblib import Parallel, delayed
from itertools import product

from tqdm import tqdm


class BayersianModel:
    '''
    There are a few key terms in bayersian statistics: 
        prior, posterior, likelyhood, evidence, inference. 
    This class tries to simulate incorporate these terms for
    the necessary computations for a bayersian inference. 

    P(theta|X) = L(theta|X) * pi(theta) / P(X)

    Attributes
    - prior: see the class Prior
    - observations 
        an (numpy) array of samples
    - log_likelyhood(theta) -> L(theta|X):
        this computes the likelyhood of a given prior
    - evidence(X)=None:
        the evidence term, which is usually cancelled in likelyhood calculation,
        default is none, and in this case it is simply 1. 

    Methods
    - log_posterior(theta) -> log P(theta|X):
    - posterior(theta) -> P(theta|X)
    - posterior_heatmap(n=(400,400),s=2,)
        returns the graph of a posterior heatmap. 
    '''

    def __init__(self,
                 prior: Prior,
                 observation,
                 log_likelyhood,
                 evidence=None):

        self.prior = prior
        self.observation = observation
        self.log_likelyhood = log_likelyhood
        self.evidence = lambda _: 1 if evidence is None else evidence

    def log_posterior(self, theta):
        return self.prior.log_pdf(theta) \
            + self.log_likelyhood(theta, self.observation) \
            - np.log(self.evidence(theta))

    def posterior(self, theta):
        return np.exp(self.log_likely(theta))

    def MPE(self, theta_range=None, n=100, njobs=None):

        if theta_range is None:
            theta_range = self.prior.theta_range

        if isinstance(n, int):
            n = np.tile(n, theta_range.shape[1])
        else:
            assert(len(n) == theta_range.shape[1])

        def compute(theta): return (
            theta, self.log_likelyhood(theta, self.observation))

        if njobs is None:
            likelyhoods = [(theta, self.log_likelyhood(theta, self.observation))
                           for theta in tqdm(list(product(
                               *[np.linspace(tr[0], tr[1], nn) for tr, nn in zip(theta_range.T, n)])))]
        else:
            likelyhoods = Parallel(n_jobs=njobs)(
                delayed(compute)(theta) for theta in product(
                    *[np.linspace(tr[0], tr[1], nn) for tr, nn in zip(theta_range.T, n)]
                ))

        return likelyhoods

    def posterior_heatmap(self, s=2, n=(200, 200), true_theta=None):
        '''
        this is only applicable if prior.dim == 2 
        n_cpus is for parallel computing
        '''

        assert(self.prior.dim == 2)

        Theta1 = np.linspace(*self.prior.theta_range[:,0], n[0])
        Theta2 = np.linspace(*self.prior.theta_range[:,1], n[1])

        XX,YY = np.meshgrid(Theta1,Theta2)
        XX = XX.flatten()
        YY = YY.flatten()
        XY = list(zip(XX,YY))

        ZZ = np.array([self.log_posterior(theta)
             for theta in tqdm(XY, desc='computing posterior')])

        plt.figure(figsize=(10, 10))

        ax = plt.axes()
        ax.set_title('the pdf for posterior')
        ax.scatter(XX, YY, c=ZZ, s=s, cmap='jet')
        self.P = np.array([XX,YY,ZZ])

        mpe_index = np.argmax(ZZ,axis=-1)
        mpe_theta = (XX[mpe_index],YY[mpe_index])
        self.mpe_theta = mpe_theta
        ax.scatter(*mpe_theta, c='black')

        if true_theta is not None:
            ax.scatter(*true_theta, c='white')
            
        return ax
