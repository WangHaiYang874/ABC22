from pyexpat import model
import numpy as np
from scipy.stats import expon


class ReactionNetwork:
    '''
    This models the stochastic network of chemical reactions, 

    TODO document later. 
    '''

    def __init__(self, reactions, species, propensity, model_parameter):
        '''
        - species: name of species.  
            arr of str 
        - reactions: descrption of input and output of each reactions.  
            arr of tuple of dict with key in chemicals and positive integer values
        - propensity(model, theta, state): 
            this is a function with input of (model, model_parameter, current_state) and the output of propensity of each reactions. 
        - model parameter:
            this should be the parameter for the propensity funciton. 
        '''

        self.chemicals = species
        self.number_of_species = len(species)
        self.index2chemical = dict(list(enumerate(species)))
        self.chemical2index = {v: k for k, v in self.index2chemical.items()}

        self.number_of_reactions = len(reactions)

        self.r_input = np.array([self._dict2vec(d) for d, _ in reactions], dtype=int)
        self.r_output = np.array([self._dict2vec(d) for _, d in reactions], dtype=int)
        self.r_diff = self.r_output - self.r_input

        self._propensity_function = propensity
        self.model_parameter = model_parameter

    def set_parameter(self, parameter):
        self.model_parameter = parameter

    def get_parameter(self):
        return self.model_parameter

    def _dict2vec(self, d):
        ret = np.zeros(self.number_of_species)
        for k, v in d.items():
            ret[self.chemical2index[k]] = v
        return ret

    def _propensity(self, model_parameter, X):
        return self._propensity_function(self, model_parameter, X)

    def propensity(self, X):
        return self._propensity(self.model_parameter, X)

    def gilespie_one_step(self, X):
        A = self.propensity(X)
        a = np.sum(A)
        dt = np.random.exponential(scale=1 / a)
        dr = np.random.choice(self.number_of_reactions, p=A / a)
        dX = self.r_diff[dr]
        return dX, dt, dr

    def gilespie(self, X, T):
        '''
        X: the population of species. 
        T: final time
        '''

        x = [X]
        t = [0]
        r = []

        while t[-1] <= T:

            dX, dt, dr = self.gilespie_one_step(x[-1])

            if t[-1] + dt > T:
                break

            x.append(dX + x[-1])
            t.append(t[-1] + dt)
            r.append(dr)

        x = np.array(x)
        t = np.array(t)
        r = np.array(r)

        return (x, r, t)

    def _one_step_tau_leaping_coupled_update_trigger(self, n, i):
        l = 1
        while i % n**l == 0:
            l += 1
        return l

    def tau_leaping_one_step_coupled(self, Xs, tau,n=2):
        
        '''
        TODO implement the case when n>2
        TODO devise appropriate return values. 
        '''
        # if n != 2 or levels != 2:
        #     print('not implemented for n!=2 yet')
        #     assert(0)

        
        levels = len(Xs)

        N = n**(levels-1)
        propensities = np.array([self.propensity(x_) for x_ in Xs])
        Xs = [[x] for x in Xs]
        # Ts = [[0] for i in range(levels)]
        
        coupled_reactions = np.zeros((levels,self.number_of_reactions), dtype=int)
        not_coupled_reactions = np.zeros((levels,self.number_of_reactions), dtype=int)

        for i in range(1,N+1):
            '''
            this loops over each time step that an event can occur and mutual 
            random variable should be generated. 

            there are a few key thing to keep in mind. 
            '''
            coupled_propensities = np.min(propensities, axis=0)
            not_coupled_propensities = propensities - coupled_propensities[np.newaxis,:]
        
            coupled_reactions = coupled_reactions + np.random.poisson(coupled_propensities*tau)[np.newaxis,:]
            not_coupled_reactions = not_coupled_reactions + np.random.poisson(not_coupled_propensities*tau)
            
            for j in range(self._one_step_tau_leaping_coupled_update_trigger(n,i)):

                Xs[j].append(Xs[j][-1] + ((not_coupled_reactions[j] + coupled_reactions[j])[:, np.newaxis]*self.r_diff).sum(axis=0))
                # Ts[j].append(Ts[j][-1] + taus[j])
                not_coupled_reactions[j] = 0
                coupled_reactions[j] = 0
                propensities[j] = self.propensity(Xs[j][-1])
        
        Xs = [x[1:] for x in Xs]
        Ts = [np.arange(1,len(Xs[j])+1)*tau*n**j for j in range(levels)]

        return Xs, Ts

    def tau_leaping_coupled(self, X, tau, T, n=2, levels=2):

        Xs = [X for i in range(levels)]
        Xs_ = [[X] for i in range(levels)]
        Ts_ = [[0] for i in range(levels)]

        t = 0
        while t < T:
            newXs_, newTs_ = self.tau_leaping_one_step_coupled(Xs, tau, n)
            Xs_ = [Xs_[j]+newXs_[j] for j in range(levels)]
            newTs_ = [tt+t for tt in newTs_]
            Ts_ = [np.concatenate((Ts_[j],newTs_[j])) for j in range(levels)]
            # I could replace this with a single concatenation 
            # it might be a bit faster but just take more memory. 
            t += tau*n**(levels-1)
            Xs = [Xs_[j][-1] for j in range(levels)]
        
        Xs_ = [np.array(x) for x in Xs_]

        return Xs_, Ts_

    def tau_leaping_one_step(self, X, tau):

        A = self.propensity(X)

        dr = np.random.poisson(tau * A)

        dX = (dr[:, np.newaxis] * self.r_diff).sum(axis=0)

        return dX, dr

    def tau_leaping(self, X, tau, T):

        x = [X]
        t = np.arange(0, T, tau)
        r = []

        for tt in t[1:]:
            dX, dr = self.tau_leaping_one_step(x[-1], tau)
            x.append(dX + x[-1])
            r.append(dr)

        x = np.array(x)
        r = np.array(r)

        return (x, r, t)

    def get_observation(self, t, x, x_mask, t_mask=None, noise=None):
        '''
        Given the time series (t,x) of the copy number of species in a simulation
        this function returns observation of the time series for species in x_mask
        at time t_mask. 

        x_mask should be an array for the name of chemical species

        t_mask should be an increase sequence of time that is in the range of t.

        noise should be a random number generator, such as noise = lambda : np.random.uniform() 
        In case of None, there is no noise. 
        
        TODO: implement noise. And Also Reaction mask. 
        '''

        i = 0
        rounded_down_t = []

        if t_mask is None:
            return x[:,[self.chemical2index[i] for i in x_mask]]

        assert(t[0] <= t_mask[0] and t[-1] >= t_mask[-1])
        for tt in t_mask:
            while t[i] <= tt and i < len(t):
                i += 1
            rounded_down_t.append(i-1)

        return x[rounded_down_t, [self.chemical2index[i] for i in x_mask]]


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

    def __init__(self, reactions, species, kinetic_rates):
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
        super().__init__(reactions, species, None, kinetic_rates)

    def _product(self, a, b):
        if b == 0:
            return 1
        if a < b:
            return 0
        else:
            return np.product(np.arange(a, a - b, -1))

    def _propensity(self, kinetic_rates, X):
        return kinetic_rates * np.array([
            np.product([self._product(a, b) for a, b in zip(X, j)])
            for j in self.r_input
        ])

    def propensity(self, X):
        return self._propensity(self.model_parameter, X)

    def gilespie_exact_log_likelihood(self, kinetic_rates, x, r, t, T):

        dt = np.diff(t)

        A = np.array([self._propensity(kinetic_rates,xx) for xx in x])
        a = np.sum(A, axis=-1)

        log_pdf_t = np.sum(expon.logpdf(dt, scale=1 / a[:-1]))
        log_pdf_r = np.sum(np.log([rr[i] for rr, i in zip(A, r)])) - np.sum(
            np.log(a))
        log_pdf_final = np.log(1 - expon.cdf(T - t[-1], scale=1 / (a[-1])))

        return log_pdf_t + log_pdf_r + log_pdf_final


class Repressilator(ReactionNetwork):
    '''
    This is an implementation for the repressilator network in biological system. 
    See [paper by Warne, et al. on arxiv](https://arxiv.org/abs/2110.14082v2) for
    description of this network. 
    '''

    def __init__(self, model_parameter):

        species = 'M1 M2 M3 P1 P2 P3'.split()
        reactions = [
            ({},            {'M1': 1}),  # mRNA transcription
            ({},            {'M2': 1}),
            ({},            {'M3': 1}),
            ({'M1': 1},      {'M1': 1, 'P1': 1}),  # protein translation
            ({'M2': 1},      {'M2': 1, 'P2': 1}),
            ({'M3': 1},      {'M3': 1, 'P3': 1}),
            ({'P1': 1},      {}),  # protein degradation
            ({'P2': 1},      {}),
            ({'P3': 1},      {}),
            ({'M1': 1},      {}),  # mRNA degradation
            ({'M2': 1},      {}),
            ({'M3': 1},      {}),
        ]
        super().__init__(reactions, species, None, model_parameter)

    def _propensity(self, theta, X):
        '''
        theta = (
            alpha0  : the leakage transcription rate
            alpha   : free_transcription rate - leakage transcription rate
            n       : hill coefficient
            K       : excluding leakage
            beta    : protein_translation and degradation rate
            gamma   : mRNA degradation rate
        )
        '''
        alpha0, alpha, n, K, beta, gamma = theta

        ret = np.zeros(self.number_of_reactions)
        # calculating the mRNA transciption propensity
        ret[:3] = alpha0 + alpha*K**n / (K**n + np.roll(X[3:], 1)**n)

        # calculating the protein translation propensity
        ret[3:6] = beta*X[:3]

        # calculating the protein degradation propensity
        ret[6:9] = beta*X[3:]

        # calculating the mRNA degradation
        ret[9:] = gamma*X[:3]

        return ret

    def propensity(self, X):
        return self._propensity(self.model_parameter, X)
