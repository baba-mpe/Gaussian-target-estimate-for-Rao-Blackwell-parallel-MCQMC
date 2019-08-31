#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 14:31:46 2019

@author: Tobias Schwedes
"""

import numpy as np
from scipy.stats import norm
from Seed import SeedGen


def rb_mp_mcqmc(x0, N, stepSize, PowerOfTwo, Stream,  mu_target, Sigma_target, WeightIn=0):
    
    """
    Implements Rao-Blackwellised multiple-proposal Markov chain
    quasi-Monte Carlo algorithm to estimate Gaussian target.
    

    Inputs:
    -------   

    x0              - array_like
                    d-dimensional array; starting value
    N               - int 
                    number of proposals per iteration
    stepSize        - float 
                    step size for proposed jump in mean
    PowerOfTwo      - int
                    defines size S of seed by S=2**PowerOfTwo-1
    Stream          - string
                    either 'cud' or 'iid'; defining what seed is used               
    """

    # Dimension
    d = len(x0)

    # Choose stream for Markoc Chain 
    xs = SeedGen(d+1, PowerOfTwo, Stream)
    
    # Set up containers for weighted sample estimates
    xI=x0
    
    # Number of iterations
    numOfIter = int(int((2**PowerOfTwo-1)/(d+1))*(d+1)/N)
    print ('Total number of Iterations = ', numOfIter)
    
    # Inverse of target covariance
    invSigma_target = np.linalg.inv(Sigma_target)      

    # Weighted Sum for weighted mean estimate
    WeightedSum = np.zeros((numOfIter,d))
    
    for n in range(numOfIter):
        
       
        # Load stream of points in [0,1]^d
        U = xs[n*(N):(n+1)*(N),:]
        
        # Sample new proposed States according to multivariate t-distribution               
        y = norm.ppf(U[:,:d], loc=np.zeros(d), scale=stepSize)
        
        # Add current state xI to proposals    
        proposals = np.insert(y, 0, xI, axis=0)
        
        # Compute Log-target probabilities        
        LogTargets = -0.5*np.dot(np.dot(proposals-mu_target, invSigma_target), \
                                 (proposals - mu_target).T).diagonal(0)
    
        # Compute Log of transition probabilities
        LogK_ni = -0.5/(stepSize**2)*np.dot(np.dot(proposals, np.identity(d)), \
                             proposals.T).diagonal(0)
        LogKs = np.sum(LogK_ni) - LogK_ni # from any state to all others
    
        # Compute weights
        LogPstates = LogTargets + LogKs
        Sorted_LogPstates = np.sort(LogPstates)
        LogPstates = LogPstates - (Sorted_LogPstates[0] + \
                np.log(1 + np.sum(np.exp(Sorted_LogPstates[1:] - Sorted_LogPstates[0]))))
        Pstates = np.exp(LogPstates)
        
        # Compute Rao-Blackwell estimate from one iteration
        WeightedStates = np.tile(Pstates, (d,1)) * proposals.T
        WeightedSum[n,:] = np.sum(WeightedStates, axis=1).copy()
        
        # Subsample from proposals to obtain new xI
        PstatesSum = np.cumsum(Pstates)
        Is = np.searchsorted(PstatesSum, U[:N-1,d:].flatten())
        xvals = proposals[Is]
        I = np.searchsorted(np.arange(1,N)/(N-1), U[N-1,d:])[0] # uniform subsampling
        xI = xvals[I,:]  
        
    # Compute Rao-Blackwell estimate
    WeightedMean = np.mean(WeightedSum, axis=0)
        
    return WeightedMean  
        
        
        
        