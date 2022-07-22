this is not a seriour readme file. it is rather the place that I keep my notes and todos for this project.

# top of my todos

- [ ] non-ABC problem. simulating the unobserved times. ABC refers to the ABC chemical problems. bridge move.
- [ ] read Overview of Approximate Bayesian Computation.
- [ ] think about how to use ml/mf for parameter estimation.
- [ ] understand how to use numba for speed up.

# other todos

- [ ] ode formulation for simulations with large copy number, need to figure out perhaps by solving the PDE with a Runge-Kutta or backward-euler etc.
- [ ] test the idea that I have today for direct sampling/fitting. This should work for a general SDE model. The idea is to discretize a SDE, generate Brownian motions on the discretization points. Plug the Brownian motions on these points. Now I have eliminated the stochasicty of the equation. I can just solve for the parameters of the equation. see [here](#parameter-inference-for-sde).
- [ ] have a better `get_observation` method in `models.py`. By now it is still somewhat imcomplete.
  - [ ] time mask,
  - [ ] reaction mask,
  - [ ] copy number mask.
- [ ] implementing the coupling between tau-leaping and gilespie.

# done

- [X] observation: transform a time series of observations into observation at fixed time plus a noise.
- [X] ❌ coupling prove on the CLT. failed attempt.
- [X] implementing the multi-level tau-leaping simulations, ABC framework for doing inference. see the notebook `tau leaping couple.ipynb`.
- [X] think about non-ABC way of doing the estimation. see note 7-18
- [X] understand coupling by reading the "Complexity ..." or Giles papers.

  - [X] mfml w 21 46 48 50
- [X] Wasserstein, coupling metrics. does not seem to be so practical in computation except for one-dimension.
- [X] nuissance variable.
- [X] have a better implementation of the coupling for levels of tau-leaping. I now have a rough working mvp. It can be better if I have the time.
- [X] send professor Goodman some notes on coupling.

# July 17th, Non-ABC

the idea of the new MCMC algorithm on chemical reaction networks has a much larger state space, that we are going to project the state space into the values we need.

Let's try to consider the following chemical reaction network with mass action kinetic and parameters $k_i$

$$
A \xLeftrightarrow[k_2]{k_1} B \xLeftrightarrow[k_4]{k_3} C
$$

Suppose the observables are $X_A(t)$ and we want to infer $k_i$. The state space of the MCMC algorithm is going to be the full chain of processes, i.e.

- $t_i$ for each time a reaction happening
- the data $X_A(t_i), X_B(t_i), X_C(t_i)$

It is unclear how to make a move in this MCMC chain, but pretend we know how. After running the MCMC chain for a real long time, we are theoretically guaranteed to have a asymptotic bias-free data. Because with this state space we can have the exact likelihood function computed.

Notice then, that for any steps to make in the MCMC chain, we need to make sure that the data $X_A(t_i)$ is consistent with our observation. This is the tricky part of this implementation. We need to have sort of a exponential bridge, in analogy to brownian bridge, to interpolate the stochastic processes. This needs to be thought through.

Professor Goodman proposed the MCMC moves to be one of the following

1. change the $k_i$, keep the reaction time fixed.
2. change the time, keep the $k_i$ fixed.  change the time contains the following moves.
   1. moving the time
   2. adding extra reactions, and therefore extra time $t_i$, with the constraint that observation is still correct.

#### Ask the following question for the new algorithm

- what problems are our algo practical to?
- compare to what other methods. ABC?
- just try some vanilla problems.

# July 9th

So far I have understood the coupling process for tau-leaping process. Might need to implement it later.

It still bugs me on how to change t

## Parameter inference for SDE

> First of all build the random variable that might be possibly taken along the path of simulation on the finest levels.
>
> For the following, I can simulate the model using this single random data at different levels. this is not a problem. Now, I can fit the models iteratively across the levels of accuracy to the observation $\mathcal D$ to get a parameter $\theta$.

I can generate one posterior sample by doing the above procedure. Each iterative step might create a large sparse matrix for me to solve. Taking another random data and do the same, I can get another sample.

This is non-MCMC and non-ABC. Could it work this way? Could it give me good enough results? Could it be computed efficiently? very likely there should be a fmm way of doing this. because this really looks like fmm.

# July 5th

For tau-leaping simulations, the parameters we tried to predict are linear. Thus if we have a close esitimate $\theta$ to the parameter $\theta^*$, we can do the simulation on parameters $\theta - \theta^*$, and then $\max_t |m_t'(\theta-\theta^*)|$ will be small because there is hardly any reactions happening inside this range. This could be a basis to define a distance of the

$$
d(\theta,\theta') = \max_t(m'_t(\theta - \theta'))
$$

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
