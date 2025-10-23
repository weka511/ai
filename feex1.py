#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (C) 2020-2025 Greenweaves Software Limited

# This is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This software is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with GNU Emacs.  If not, see <http://www.gnu.org/licenses/>

'''
   Exercise 1, posterior probabilities, from A tutorial on the free-energy
   framework for modelling perception and learning, by Rafal Bogacz.
   Write a computer program that computes the posterior probabilities of
   sizes from 0.01 to 5, and plots them.
'''

from os.path import join
from pathlib import Path
from matplotlib.pyplot import figure, show
import numpy as np
from scipy.integrate import quad
from scipy.stats import norm

def get_likelihood(u,v,Sigma_u=1, g=lambda v:v**2):
    '''
    Calculate Likelihood

    Parameters:
        u        Light intensity
        v        Size of a food item
        Sigma_u  Expected standard deviation for u
        g        Function that relates light intensity with size
    '''
    return norm(g(v),Sigma_u).pdf(u)

def get_prior(v, vp=3, Sigma_p=1):
    '''
    Animal expects size to be normally distributed

    Parameters:
        v       Size of a food item
        vp      Prior - expected value of v
        Sigma_p Expected standard deviation
    '''
    return norm(vp, Sigma_p).pdf(v)

def get_evidence(u, minsize = 0,maxsize = 5,epsabs=0.0001):
    '''
    Calculate Evidence by integrating get_prior*get_likelihood

    Parameters:
        u        Light intensity
        epsabs   Absolute error tolerance
    '''
    return quad(lambda v:get_prior(v)*get_likelihood(u,v),minsize,maxsize,epsabs=epsabs)[0]

def get_posterior(v,u,minsize = 0,maxsize = 5):
    '''
    Calculate Posterior probability for size

    Parameters:
        u        Light intensity
        v        Size of a food item
    '''
    return get_prior(v)*get_likelihood(u,v)/get_evidence(u,minsize = minsize,maxsize = maxsize)

def get_max_posterior(Sizes,Posterior):
    '''
    Calculate maximum posterior proability

    Parameters:
        Sizes      A collection of sizes
        Posterior  Posterior probability for each size
    '''
    index_max = np.argmax(Posterior)
    return Sizes[index_max],Posterior[index_max]

if __name__ == '__main__':
    u = 2   # Observed light intensity
    minsize = 0
    maxsize = 5
    Sizes = np.linspace(minsize,maxsize,num=500)
    Posterior = get_posterior(Sizes,u,minsize = minsize,maxsize = maxsize)
    size,max_posterior = get_max_posterior(Sizes,Posterior)
    fig = figure(figsize=(10,10))
    ax = fig.add_subplot(1,1,1)
    ax.scatter(Sizes,Posterior, s = 1, c = 'xkcd:blue', label = 'Posterior probability of sizes')
    ax.set_xlabel('v')
    ax.set_ylabel('p(v|u)')
    ax.vlines(size,0,max_posterior, colors = 'xkcd:red', linestyles = 'dotted', label = f'Max posterior occurs at {size:.2f}')
    ax.legend()
    ax.set_title('Bogacz, Exercise 1')
    fig.savefig(join('figs',Path(__file__).stem))
    show()
