from pyexpat import model
import numpy as np
from scipy.stats import expon



class ReactionNetwork:
    '''
    This models the stochastic network of chemical reactions, 
    
    # the paramters
    - number of species = N
    - number of reactions = M
    - reactions stochio :has: shape(M,2,N)
    - kinetic rates :has: shape(M)
    
    # usage:
    TODO document later. 
    '''
    
    def __init__(
        self,
        reactions,
        species, 
        propensity,
        model_parameter
    ):
        '''
        - species: name of species.  
            arr of str 
        - reactions: descrption of input and output of each reactions.  
            arr of tuple of dict with key in chemicals and positive integer values
        - propensity(model, theta,state): 
            this is a function with input of (model, model_parameter, current_state) and the output of propensity of each reactions. 
        - model parameter:
            this should be the parameter for the propensity funciton. 
        '''
        
        self.chemicals = species
        self.number_of_species = len(species)
        self.index2chemical = dict(list(enumerate(species)))
        self.chemical2index = {v:k for k,v in self.index2chemical.items()}
        
        self.number_of_reactions = len(reactions)
        
        self.r_input = np.array([self._dict2vec(d) for d,_ in reactions])
        self.r_output = np.array([self._dict2vec(d) for _,d in reactions])
        self.r_diff = self.r_output - self.r_input
        
        self._propensity = lambda theta, X: propensity( self, theta, X)
        self.model_parameter = model_parameter
        
    def set_parameter(self, parameter):
        self.model_parameter = parameter

    def get_parameter(self):
        return self.model_parameter

    def _dict2vec(self,d):
        ret = np.zeros(self.number_of_species)
        for k,v in d.items():
            ret[self.chemical2index[k]] = v
        return ret
    
    def propensity(self,X):
        return self._propensity(self.model_parameter,X)
    
    def gilespie_one_step(self,X):
        A = self.propensity(X)
        a = np.sum(A)
        dt = np.random.exponential(scale=1/a)
        dr = np.random.choice(self.number_of_reactions,p=A/a)
        dX = self.r_diff[dr]
        return dX,dt,dr
    
    def gilespie(self,X,T):
        
        '''
        X: the population of species. 
        T: final time
        '''
        
        x = [X]
        t = [0]
        r = []
        
        while t[-1] <= T:
            
            dX,dt,dr = self.gilespie_one_step(x[-1])
            
            if t[-1] + dt > T: break
            
            x.append(dX+x[-1])
            t.append(t[-1]+dt)
            r.append(dr)
        
        return (x,r,t)
        
    def tau_leaping_one_step(self,X,tau):

        A = self.propensity(X)

        dr = np.random.poisson(tau*A)

        dX = (dr[:,np.newaxis]*self.r_diff).sum(axis=0)

        return dX,dr
    
    def tau_leaping(self,X,tau,T):
        
        x = [X]
        t = np.arange(0,T,tau)
        r = []
        
        for tt in t:
            dX, dr = self.tau_leaping_one_step(x[-1],tau)            
            x.append(dX+x[-1])
            r.append(dr)
        
        x = np.array(x)
        r = np.array(r)
        
        return (x,r,t)
    

class ChemicalReactionNetwork(ReactionNetwork):
    '''
    This simulates the chemical reaction networks with the 
    propensity given by the **mass action kinetics**. 
    
    # the paramters
    - number of species = N
    - number of reactions = M
    - reactions stochio :has: shape(M,2,N)
    - kinetic rates :has: shape(M)
    
    # usage:
    TODO document later. 
    '''
    
    def __init__(
        self,
        reactions,
        species,
        kinetic_rates
        ):
        '''
        - species: name of species.  
            arr of str 
        - reactions: descrption of input and output of each reactions.  
            arr of tuple of dict with key in chemicals and positive integer values
        - propensity(theta,state): 
            this is a function with input of (model_parameter, current_state) and the output of propensity of each reactions. 
        - kinetic_rates: 
            the parameters for the propensity function
        '''
        
        super.__init__(self,reactions,species,propensity=None,model_parameter=kinetic_rates)
        
    def product(self,a,b):
        if b == 0:
            return 1
        if a < b:
            return 0
        else:
            return np.product(np.arange(a,a-b,-1))
    
    def _propensity(self, kinetic_rates, X):
        return kinetic_rates * np.array([np.product([self.product(a,b) for a,b in zip(X,j)]) for j in self.r_input])
        
    def propensity(self,X):
        return self._propensity(self.model_parameter,X)
    
    def gilespie_exact_log_likelihood(self,kinetic_rates,x,r,t,T):
        
        dt = np.diff(t)
        
        A = np.array([self.propensity_(xx,kinetic_rates) for xx in x])
        a = np.sum(A,axis=-1)
        
        log_pdf_t = np.sum(expon.logpdf(dt,scale=1/a[:-1]))
        log_pdf_r = np.sum(np.log([rr[i] for rr,i in zip(A,r)])) - np.sum(np.log(a))
        log_pdf_final = np.log(1-expon.cdf(T-t[-1],scale=1/(a[-1])))
        
        return log_pdf_t + log_pdf_r + log_pdf_final
    
    