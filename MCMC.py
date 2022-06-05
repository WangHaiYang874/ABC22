import numpy as np
from scipy import stats as ss
from BayersianModel import BayersianModel
from tqdm import tqdm


class NaiveSampler:
    def __init__(self, model:BayersianModel, burn_in=100):
        self.model = model
        self.burn_in = burn_in
        
    def get_init_position(self):
        '''
        returns a random initial position for the MCMC
        '''
        
        return self.model.prior.sampler()
    
    def proposal(self, theta, theta_range='free', box_size=.1):
        '''
        return theta, log p(theta)
        
        this propose a new theta based on the current theta. 
        it basically generates a small box around theta of radius box_size
        and then uniformly pick a point from this box. 
        
        in most cases, theta can move freely, yet in some other cases, 
        theta has some restrictions: for example, must be positive, etc. 
        
        I have only implemented three cases, 'free', 'positive', or a list of 
        intervals that defines the range of theta. 
        '''
        
        t1,t2 = theta
        
        if theta_range=='free': 
            t1rg = (t1-box_size,t1+box_size)
            t2rg = (t2-box_size,t2+box_size)
        elif theta_range=='positive':
            t1rg = (max(t1-box_size,0),t1+box_size)
            t2rg = (max(t2-box_size,0),t2+box_size)
        else:
            print("Error unimplemented")
            assert(False)
        l1 = t1rg[1] - t1rg[0]
        l2 = t2rg[1] - t2rg[0]
        a = l1*l2
        log_p = - np.log(a)
        
        t1 = np.random.uniform(*t1rg)
        t2 = np.random.uniform(*t2rg)
        return (t1,t2), log_p
    
    def simulation(self,n=10000,theta_range='free',box_size=.1):
        theta = self.get_init_position()
        samples = [theta]
        for i in tqdm(range(n+self.burn_in), desc='sampling posterior'):
            theta_,log_t = self.proposal(theta)
            log_posterior = self.model.log_posterior(theta_)\
                - self.model.log_posterior(theta)
            log_p = log_t + log_posterior
            if log_p >= 0:
                theta = theta_
            if log_p <= np.log(np.random.uniform()):
                theta = theta_
            samples.append(theta)
        return np.array(samples[-n:])