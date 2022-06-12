import numpy as np
from scipy import stats as ss
from BayersianModel import BayersianModel
from tqdm import tqdm


class NaiveSampler:
    def __init__(self, model: BayersianModel, burn_in=100):
        self.model = model
        self.burn_in = burn_in

    def get_init_position(self, constraint=None):
        '''
        returns a random initial position for the MCMC
        '''

        if constraint is None:
            return self.model.prior.sampler()
        else:
            theta = self.model.prior.sampler()
            while not constraint(theta):
                theta = self.model.prior.sampler()
            return theta

    def get_box(self, theta, box_size, constraints=None):

        box = []
        volume = 1

        if constraints is None:
            for t in theta:
                lower = t-box_size
                upper = t+box_size
                box.append((lower, upper))
                volume *= (upper-lower)
        else:

            if len(theta) != len(constraints):

                print('\n---------------\n')
                print(theta, constraints, sep='\n')
                print('\n---------------\n')
                assert(False)

            for t, constraint in zip(theta, constraints):
                if constraint == None:
                    lower = t-box_size
                    upper = t+box_size
                else:
                    a, b = constraint
                    lower = max(a, t-box_size)
                    upper = min(b, t+box_size)
                box.append((lower, upper))
                volume *= (upper-lower)

        box = np.array(box).T

        return box, volume

    def proposal(self, theta, theta_range=None, box_size=.1):
        '''
        return theta, log p(theta)

        this propose a new theta based on the current theta. 
        it basically generates a small box around theta of radius box_size
        and then uniformly pick a point from this box. 

        in most cases, theta can move freely, yet in some other cases, 
        theta has some restrictions: for example, must be positive, etc. 

        I have only implemented the following cases, 
            - 'free': the parameter can go anywhere
            - 'box': given a n-d box as the constraint for next parameter
            - todo 'constraint': give a function that defines a half space.. 
            - 'positive': simply sample the positive parameters

        Possible bugs: 
            - the `contraint` mode might create a (hopefully small) bias.     
        '''

        if theta_range == 'positive':
            theta_range = [(0, np.inf) for _ in theta]
        if theta_range == 'constraint':
            raise('todo')

        box, a = self.get_box(theta, box_size, theta_range)
        new_theta = np.random.uniform(*box)

        _, a_ = self.get_box(new_theta, box_size, theta_range)

        return new_theta, np.log(a/a_)

    def simulation(self, n=10000, theta_range=None, box_size=.1):

        theta = self.get_init_position()
        samples = [theta]

        for i in tqdm(range(n+self.burn_in), desc='sampling posterior'):

            theta_, log_t = self.proposal(theta, theta_range=theta_range,
                                          box_size=box_size)

            # if not constraint(theta):
            #     print('wtf')
            #     assert(False)
            # if not constraint(theta_):
            #     theta_ = theta
            #     pass

            log_posterior = self.model.log_posterior(theta_)\
                - self.model.log_posterior(theta)
            log_acceptance = log_t + log_posterior

            if log_acceptance >= 0:
                theta = theta_
            if np.log(np.random.uniform()) <= log_acceptance:
                theta = theta_
            else:
                # reject
                theta = theta

            samples.append(theta)

        return np.array(samples)
