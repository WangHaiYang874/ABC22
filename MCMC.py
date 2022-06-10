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
    
    def proposal(self, theta, theta_range='free', box_size=.1, constraint=None):
        '''
        return theta, log p(theta)
        
        this propose a new theta based on the current theta. 
        it basically generates a small box around theta of radius box_size
        and then uniformly pick a point from this box. 
        
        in most cases, theta can move freely, yet in some other cases, 
        theta has some restrictions: for example, must be positive, etc. 
        
        I have only implemented the following cases, 
            - 'free': the parameter can go anywhere
            - 'positive': the parameters must be positive
            - 'constraint': 
                it needs to be supplemented with an input contraint,
                which is a function of theta, that returns true or false. 
                sample as in the 'free', 
                but if constraint(theta) = False,
                return -np.inf
            - 'box':
                unimplemented yet
        
        Possible bugs: 
            - the `contraint` mode might create a (hopefully small) bias.     
        '''
        
        d = len(theta)
        
        if theta_range in ['free','constraint']: 
            Theta_range = np.array([(i-box_size,i+box_size) for i in theta]).T
        elif theta_range=='positive':
            Theta_range = np.array([(max(i-box_size,0),i+box_size) for i in theta]).T
        elif theta_range=='box':
            print('unimplemented')
            assert(False)
        else:
            print('unimplemented')
            assert(False)
        
        a1 = np.prod(Theta_range[:,1] - Theta_range[:,0])
        log_p = - np.log(a1)
        new_theta = np.random.uniform(*Theta_range)
        
        if theta_range in ['free','constraint']: 
            log_p += d*np.log(2*box_size)
        elif theta_range == 'positive':
            log_p += np.prod([i + box_size - max(i-box_size,0) for i in new_theta])
        elif theta_range == 'box':
            print('unimplemented')
            assert(False)
        else:
            print('unimplemented')
            assert(False)
        
        if theta_range == 'constraint' and not constraint(new_theta):
            log_p = -np.inf
        
        return new_theta, log_p
    
    def simulation(self,n=10000,theta_range='free',box_size=.1,constraint=None):
        theta = self.get_init_position()
        samples = [theta]
        for i in tqdm(range(n+self.burn_in), desc='sampling posterior'):
            theta_,log_t = self.proposal(theta,theta_range=theta_range,
                                         constraint=constraint,box_size=box_size)
            if constraint(theta_):
                log_posterior = self.model.log_posterior(theta_)\
                    - self.model.log_posterior(theta)
                log_p = log_t + log_posterior
                if log_p >= 0:
                    theta = theta_
                if log_p <= np.log(np.random.uniform()):
                    theta = theta_
            else:
                theta = theta_
            samples.append(theta)
        return np.array(samples[-n:])