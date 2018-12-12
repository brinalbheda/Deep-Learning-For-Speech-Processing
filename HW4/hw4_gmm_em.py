#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 8 00:30:37 2018

@author: brinalbheda
"""

import numpy as np

# Generate the random data     
np.random.seed(0)
mean = [1, 4, 7]
Sigma = [.1, .3, .5]
n = [500, 500, 500]
tot_n = np.sum(n)
gaussian = []

for i in range(len(mean)):
    data = Sigma[i] * np.array(np.random.randn(n[i], 1)) + mean[i]
    gaussian.append(data)
X = np.array(gaussian).reshape(tot_n, 1)


class GMM:

    def __init__(self, n_clusters, n_data, max_iterations, tot_n):
        self.n_clusters = n_clusters 
        self.n_data = n_data 
        self.max_iterations = max_iterations 
        self.tot_n = tot_n 
        
    def get_likelihood(self, X, mu, Sigma, pi):
        temp1 = (X-mu) * 1/Sigma
        temp = temp1 * (X-mu)
        const = pi/np.sqrt(2*3.14*np.abs(Sigma))
        likelihood = (const)*np.exp(-0.5*temp)
        likelihood = np.sum(likelihood, axis=1)

        return likelihood
    
        
    def fit_EM(self, X):
        mu = np.array([4, 6, 9])        
        Sigma = np.array([.2, .21, 0.3])        
        pi = (self.n_data/np.sum(self.n_data)).reshape(self.n_clusters, 1)
        

        for itr in range(self.max_iterations):
            #E-step
            prob = []
            gamma = []
            for k in range(self.n_clusters):
                likelihood = self.get_likelihood(X, mu[k], Sigma[k], pi[k])
                prob.append(likelihood)
            likelihoods = np.array(prob)   
            gamma = likelihoods
            denom =np.sum(likelihoods,axis=0).reshape(1,likelihoods.shape[1])
            gamma = gamma/denom
            
            #M-step
            self.n_data = np.sum(gamma, axis=1)
            pi = self.n_data/self.tot_n
            mu = np.dot(gamma, X)
            for k in range(self.n_clusters):
                if pi[k]!=0:
                    mu[k] = mu[k]/self.n_data[k]
                    te = np.multiply((X-mu[k]), (X-mu[k]))
                    Sigma = np.dot(gamma, te)
                    Sigma[k] = Sigma[k]/self.n_data[k]
                  
        return mu, Sigma, pi


gmm = GMM(len(mean), n, 50, tot_n)
mu, sig, pi = gmm.fit_EM(X)

print("Actual Mean: ", mean)
print("Estimated Mean using GMM: ", mu)
print("")
print("Actual Variance: ", Sigma)
print("Estimated Variance using GMM: ", sig)
print("")

