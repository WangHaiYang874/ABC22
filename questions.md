# top of my todos

- [ ] implementing the multi-level tau-leaping simulations, ABC framework for doing inference, and then the 
- [ ] think about non-ABC way of doing the estimation. 
- [ ] observation: transform a time series of observations into observation at fixed time plus a noise. 


# other todos 

- [ ] ode formulation for simulations with large copy number, need to figure out perhaps by solving the PDE with a Runge-Kutta or backward-euler etc.  

# July 9th

So far I have understood the coupling process for tau-leaping process. Might need to implement it later. 

It still bugs me on how to change t

## todo 
- [x] understand coupling by reading the "Complexity ..." or Giles papers.
   - [x] mfml w 21 46 48 50
- [x] Wasserstein, coupling metrics. does not seem to be so practical in computation. 
- [ ] test the idea that I have today for direct sampling/fitting. this needs me to understand the coupling process better
- [ ] nuissance variable. 

## Brownian bridge and inverse problem from the tau-leaping and coupling perspectives. 

>First of all build the random variable that might be possibly taken along the path of simulation on the finest levels. 
> 
> For the following, I can simulate the model using this single random data at different levels. this is not a problem. Now, I can fit the models iteratively across the levels of accuracy to the observation $\mathcal D$ to get a parameter $\theta$. 


I can generate one posterior sample by doing the above procedure. Each iterative step might create a large sparse matrix for me to solve. Taking another random data and do the same, I can get another sample. 

This is non-MCMC and non-ABC. Could it work this way? Could it give me good enough results? Could it be computed efficiently? very likely there should be a fmm way of doing this. because this really looks like fmm. 

# July 5h

For tau-leaping simulations, the parameters we tried to predict are linear. Thus if we have a close esitimate $\theta$ to the parameter $\theta^*$, we can do the simulation on parameters $\theta - \theta^*$, and then $\max_t |m_t'(\theta-\theta^*)|$ will be small because there is hardly any reactions happening inside this range. This could be a basis to define a distance of the 

$$d(\theta,\theta') = \max_t(m'_t(\theta - \theta'))$$  

Now, notice that in a MRT step, I do not know the $\theta^*$ but only the proposed $\theta'$ and current $\theta$. And clearly we are more inclined to have a 
$$
d(\theta',\theta^*) < d(\theta,\theta^*) 
$$

This idea might evetually lead to some gradient descent way of giving a proposal function, for example to let $m(\theta') = m(\theta + d\theta) \approx m(\theta) + m(d\theta)$ become away of proposing. the formular is incorrect, however, comparing form the standard calculus formular of of $m(\theta') = m(\theta) + m'(\theta) d\theta$. 

Why might this work? At the point $\theta$, the function $m'(\theta)d\theta$ and $m(d\theta)$ might be proportional. This is certainly the case for polynomial functions. 

And if this one works, what's so great is that $m(d\theta)$ can be computed in a mere preconditioning process very efficiently. 

Might this method work? I will check it out another day. I cannot come up with good thoughts now and better do my brain-less homework. 



# June 27th

1. where is the canonical place to put my texts. Maybe I should just have a function that prints out the text after each plot. Or I can input the texts into the model class...
   1. more information in plots (MCMC run length, among others).
2. do one of warne's experiement. this requires a frame work of **rejections** and everything. 
   1. have more than one kind of atom (not just H and H2) and measure the copy number of just some of the species.
3. affine inv with 2 pts moving at the same time. 

## optimization works
1. I should clean up my codes in `BayesianModel.py` and `models.py`, there are a few terms that should be simplified. And `models.py` should be a sub-class of `BayesianModel.py`. 

## thoughts

其实越是高维度的东西就越是容易anisotropic的. 这一点在mcmc中如何体现呢. 


# June 21st

1. canonical way to find the mode of samples from continuous r.v. 
   the mpe
2. affine inv with multiple points moving. 
3. population mcmc and etc. 