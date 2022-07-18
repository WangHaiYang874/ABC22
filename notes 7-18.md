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
2. change the time, keep the $k_i$ fixed. 

