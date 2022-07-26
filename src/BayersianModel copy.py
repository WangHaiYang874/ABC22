import Prior
import numpy as np
from matplotlib import pyplot as plt
from joblib import Parallel, delayed
from itertools import product
import math
from tqdm import tqdm

class SimpleBayesianModel:
    '''
    
    This is a simplified version of the Bayesian probablistic model. 
    
    Attributes:
     - p = log_prob(theta|observation):
        a function that returns log_prob of theta given observation. 
     - observation:
        a list of individual samples. 
        Preferably a numpy array of shape (N,...)    
    
    # Mumbling. 
    Given you a model and parameters for the model, the model can generate
    some output. For example, given you the 
    - logistic model, 
    - parameters of death rate, birth rate, and initial population
    - one can compute the population size in the future. 
    Many times in science, we know the models by scientific principles, 
    we know the output by experiments, it is only left for us to know 
    the parameters so that we can know the full answer. 
    
    And this is what Bayersian stats is for. 
    
    The model is described by a probability distribution 
        p(observation|theta)
    And the obeservation is given, using the Bayes rule
        p(theta|Observaton) p(observation) = p(Observation|theta) p(theta)
    One can know what's our parameter should be. 
    
    '''

    def __init__(self, observation, log_prob):

        self.observation = observation
        self.log_prob = lambda theta: log_prob(theta, self.observation)

    def prob(self, theta):
        return np.exp(self.log_likely(theta))

    def grid_compute(self, theta_range,precision=.1):
        '''
        theta_range: 
            indicates where each parameter should take. 
            should be a numpy array.  
                theta_range = [
                    [a_1,a_2, ... , a_n], 
                    [b_1,b_2, ... , b_n]
                ]
            where n is the dimention of parameter.
            and (a_i,b_i) is the range for parameter of dimension n.
        precision: 
            in each dimension, the near-by equally spaced pts between (a_i,b_i)
            has distance smaller than presicion
        '''
        
        Ns = [math.ceil(i) for i in (theta_range[1]-theta_range[0])/precision]
        Thetas = [np.linspace(interval[0],interval[1],n) for n,interval in zip(Ns,theta_range.T)]
        Thetas = [i.flatten() for i in np.meshgrid(*Thetas)]
        
        result = np.array([self.log_prob(theta) for theta in tqdm(Thetas)])
        
        Thetas = Thetas.T
        
        
        self.P = np.array(*(Thetas), result)
                
        return result, *Thetas
            
    
    def posterior_heatmap(self, theta_range, true_theta=None, precision=.1,s=1):
        '''
        this is only applicable if prior.dim == 2 
        '''

        assert(len(theta_range.T) == 2)
        
        ZZ,XX,YY = self.grid_compute(theta_range, precision) 
        
        plt.figure(figsize=(15, 15))

        ax = plt.axes()
        ax.set_title('the pdf for posterior')
        ax.scatter(XX, YY, c=ZZ, s=s, cmap='jet')

        mpe_index = np.argmax(ZZ,axis=-1)
        mpe_theta = (XX[mpe_index],YY[mpe_index])
        self.mpe_theta = mpe_theta
        ax.scatter(*mpe_theta, c='black')

        if true_theta is not None:
            ax.scatter(*true_theta, c='white')