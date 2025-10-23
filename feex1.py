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

from matplotlib.pyplot import figure, show
import numpy as np
from scipy.integrate import quad
from scipy.stats import norm

def p_u_v(u,v,Sigma_u=1, g=lambda v:v**2):
    '''
    Calculate Likelihood

    Parameters:
        u
        v
        Sigma_u
        g
    '''
    return norm(g(v),Sigma_u).pdf(u)

def p_v(v, vp=3, Sigma_p=1):
    '''
    Prior expectation of size

    Parameters:
        v
        vp
        Sigma_p
    '''
    return norm(vp, Sigma_p).pdf(v)

def p_u(u):
    '''
    Calculate Evidence by integrating p_v*p_u_v
    '''
    return quad(lambda v:p_v(v)*p_u_v(u,v),0,5,epsabs=0.0001)[0]

def get_posterior(v,u):
    '''
    Calculate Posterior probability
    '''
    return p_v(v)*p_u_v(u,v)/p_u(u)

def get_max_posterior(Posterior,Sizes):
    '''
    Calculate maximum posterior proability
    '''
    index_max = np.argmax(Posterior)
    return Sizes[index_max],Posterior[index_max]

if __name__ == '__main__':
    u = 2   # Observed light intensity
    Sizes = np.linspace(0,5,num=500)
    evidence = p_u(u)
    Posterior = get_posterior(Sizes,u)
    x,y = get_max_posterior(Posterior,Sizes)
    fig = figure(figsize=(10,10))
    ax = fig.add_subplot(1,1,1)
    ax.scatter(Sizes,Posterior, s = 1, c = 'xkcd:blue', label = 'Posterior probability of sizes')
    ax.set_xlabel('v')
    ax.set_ylabel('p(v|u)')
    ax.vlines(x,0,y, colors = 'xkcd:red', linestyles = 'dotted', label = f'Max posterior={x:.2f}')
    ax.legend()
    ax.set_title('Bogacz, Exercise 1')
    fig.savefig('figs/feex1')
    show()
