import numpy as np
from scipy.stats import expon


class ChemicalReactionNetwork:
    '''
    This models the stochastic network of chemical reactions, 
    
    # the paramters
    - number of species = N
    - number of reactions = M
    - reactions stochio :has: shape(M,2,N)
    - kinetic rates :has: shape(M)
    
    # the methods
    - __init__
    - 
    '''
    
    def __init__(
        self,
        kinetic_rates,
        reactions,
        chemicals, 
    ):
        '''
        - kinetic_rates: shape (N)
        - chemicals: arr of str, len(arr) = N
        - reactions: arr of tuple of dict with key in chemicals and positive integer values
        '''
        
        self.chemicals = chemicals
        self.N = len(chemicals)
        self.index2chemical = dict(list(enumerate(chemicals)))
        self.chemical2index = {v:k for k,v in self.index2chemical.items()}
        
        self.M = len(reactions)
        
        self.r_input = np.array([self.dict2vec(d) for d,_ in reactions])
        self.r_output = np.array([self.dict2vec(d) for _,d in reactions])
        self.r_diff = self.r_output - self.r_input
        self.r_k_rates = np.array(kinetic_rates)
        
    def dict2vec(self,d):
        ret = np.zeros(self.N)
        for k,v in d.items():
            ret[self.chemical2index[k]] = v
        return ret
    
    def product(self,a,b):
        if b == 0:
            return 1
        if a-b+1 < 1:
            return 0
        else:
            return np.product(np.arange(a,a-b,-1))
    
    def propensity(self,X):
        return self.r_k_rates * np.array([np.product([self.product(a,b) for a,b in zip(X,j)]) for j in self.r_input]) 
    
    def propensity_(self,X,kinetic_k_rates):
        return kinetic_k_rates * np.array([np.product([self.product(a,b) for a,b in zip(X,j)]) for j in self.r_input]) 
    
    def gilespie1(self,X):
        A = self.propensity(X)
        a = np.sum(A)
        dt = np.random.exponential(scale=1/a)
        dr = np.random.choice(self.M,p=A/a)
        dX = self.r_diff[dr]
        return dX,dt,dr
    
    def gilespie(self,X,T):
        x = [X]
        t = [0]
        r   = []
        
        while t[-1] < T:
            dX,dt,dr = self.gilespie1(x[-1])
            x.append(dX+x[-1])
            t.append(t[-1]+dt)
            r.append(dr)
        
        # a.s. t[-1] >= T, so we pop out the last term. 
        
        x = np.array(x[:-1])
        t = np.array(t[:-1])
        r = r[:-1]
        
        return (x,r,t)
            
    def set_kinetic_rates(self,kinetic_rates):
        self.r_k_rates = kinetic_rates
    
    def get_kinetic_rates(self):
        return self.r_k_rates
    
    def tau_leaping1(self,X,tau):
        A = self.propensity(X)
        dX = (np.random.poisson(tau*A)[:,np.newaxis]*self.r_diff).sum(axis=0)
        return tau, dX
    
    def log_likelihood(self,kinetic_rates,x,r,t,T):
        
        dt = np.diff(t)
        
        A = np.array([self.propensity_(xx,kinetic_rates) for xx in x])
        a = np.sum(A,axis=-1)
        
        log_pdf_t = np.sum(expon.logpdf(dt,scale=1/a[:-1]))
        log_pdf_r = np.sum(np.log([rr[i] for rr,i in zip(A,r)])) - np.sum(np.log(a))
        log_pdf_final = np.log(1-expon.cdf(T-t[-1],scale=1/(a[-1])))
        
        return log_pdf_t + log_pdf_r + log_pdf_final
    