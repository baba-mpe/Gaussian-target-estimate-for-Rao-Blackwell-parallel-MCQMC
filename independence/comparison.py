#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 14:46:35 2019

@author: Tobias Schwedes
"""

import numpy as np
from metropolis_hastings import metropolis
from rao_blackwell_mp_mcqmc import rb_mp_mcqmc


#########################
# Simulation parameters #
#########################

# General
d=10 # dimension
mu_target = np.zeros(d)         # target mean
Sigma_target = np.identity(d)   # target covariance
numOfSim = 500           # number of simulations

print ("Dimension = ", d)
print ("Number of simulations = ", numOfSim)

# Metropolis-Hastings
n_mh = 32768           # number of samples
stepSize = 2.05 # RW: 1.7(d=5); 1.5(d=6); 1.35(d=7); 1.25(d=8); 1.15 (d=9); 1.05(d=10)
                # Independence: 2.3 (d=5); 2.1 (d=6); 1.95 (d=7); 1.85 (d=8); 2. (d=9); 2.05 (d=10)
x0 = np.zeros(d)
mus_mh = np.zeros((numOfSim,d))

# Rao-Blackwell MP-MCQMC
N = 512          # number of proposals
PowerOfTwo = 15  # 2**PowerOfTwo = n_mh for comparison
Stream = 'cud'
mus_rb = np.zeros((numOfSim,d))

print ("Number of proposals = ", N)

##################
# Simulation run #
##################

for i in range(numOfSim):
    
#    # Metropolis-Hastings 
#    output_mh = metropolis(x0, n_mh, stepSize, mu_target, Sigma_target)
#    mus_mh[i,:] = np.mean(output_mh[0],axis=0)
#    print (output_mh[1])
#    
    # Rao-Blackwell MP-MCQMC
    mus_rb[i,:] = rb_mp_mcqmc(x0, N, stepSize, PowerOfTwo, Stream,  mu_target, Sigma_target)


# Effective sample sizes for 
ess_mh = np.mean(Sigma_target.diagonal(0)/np.var(mus_mh,axis=0)) # Metropolis-Hastings
ess_rb = np.mean(Sigma_target.diagonal(0)/np.var(mus_rb,axis=0)) # RB-MP-MCQMC


print ("ESS Metropolis-Hastings = ", ess_mh)
print ("ESS Rao-Blackwell MP-MCQMC = ", ess_rb)
