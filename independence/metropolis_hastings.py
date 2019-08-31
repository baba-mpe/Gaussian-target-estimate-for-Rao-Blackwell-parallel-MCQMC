#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 12:15:43 2019

@author: Tobias Schwedes
"""

import numpy as np
#from scipy.stats import multivariate_normal



def metropolis(x0, n, stepSize, mu_target, Sigma_target, BurnIn=0):
    
    """
    Function to implement Metropolis-Hastings to estimate Gaussian target.

    Inputs:
    -------  
    
    x0          - array_like
                initial point of the chain
    n           - int
                number of iterations
    StepSize    - float
                step size of Gaussian proposal kernel

    Outputs:
    -------  
    xvals       - array_like
                samples
    arate       - float
                acceptance rate
    """

    # Dimension
    d = len(x0)

    # Set up containers for samples and acceptance values
    xvals = np.zeros((n+1,d)); xvals[0,:] = x0
    x = x0 # initial value
    acpt = list()

    # Inverse of target covariance
    InvSigma_target = np.linalg.inv(Sigma_target)  
    
    # Log target of current point
    currentLogTarget = -0.5*np.dot(x-mu_target,np.dot(InvSigma_target,x-mu_target))
    
    for i in range(n):

        # Proposed point (independence sampler)
        xNew = np.dot(np.random.normal(0,1,d), stepSize*np.identity(d))

        # Proposed target
        proposedLogTarget = -0.5*np.dot(xNew-mu_target,np.dot(InvSigma_target,xNew-mu_target))
        
        # Accept/reject step
        ratio = proposedLogTarget - currentLogTarget
        if (ratio > 0) or (ratio > np.log(np.random.uniform())):
            currentLogTarget = proposedLogTarget
            x = xNew
            acpt.append(1.)

        # add new sample
        xvals[i+1,:] = x
    
    # Acceptance rate
    arate = sum(acpt)/(n+1)

    return xvals, arate 
    