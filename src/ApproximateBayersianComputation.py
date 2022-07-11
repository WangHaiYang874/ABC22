'''
author: Haiyang Wang 
email: hw1927@nyu.edu

This file intends to provide the general framework of computing the 
likelyhood when closed form expression does not exist. 

The idea can be expressed by the formula

p(theta|Y_obs) 
\simeq p(theta|rho(Y_obs,Y_s) < epsilon) 
\propo P(rho(Y_obs,Y_s)<epsition | theta) p(theta)

'''
