#!/usr/bin/env python

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
   Exercise 3--neural implementation-- from A tutorial on the free-energy
   framework for modelling perception and learning, by Rafal Bogacz
'''

from os.path import join
from pathlib import Path
from matplotlib.pyplot import figure, show
from matplotlib import rc
import numpy as np

def g(v):
    '''
    Model for intensity of light

    Parameters:
        v   Food size
    '''
    return v**2

def g_prime(v):
    '''
    Calculate derivative of g

    Parameters:
        v   Food size
    '''
    return 2*v

if __name__ == '__main__':
    v_p  = 3   # Mean of prior for food size
    Sigma_p = 1   # Variance of prior
    Sigma_u = 1   # Variance of sensory noise
    u = 2   # Observed light intensity
    T = 501
    phi = v_p  # Estimate for food size
    epsilon_p = 0    # prediction error food size
    epsilon_u = 0    # prediction error sensory input
    dt = 0.01

    phis = np.zeros((T))       # Estimates for food size
    epsilon_us = np.zeros((T)) # Prediction error for light intensity
    epsilon_ps = np.zeros((T)) # Prediction error for food size
    phis[0] = phi
    TT = np.arange(0,5+dt,dt)
    for t in range(1,T):
        phi_dot = epsilon_u*g_prime(phi) - epsilon_p
        epsilon_p_dot = phi - v_p - Sigma_p*epsilon_p
        epsilon_u_dot = u - g(phi) - Sigma_u*epsilon_u

        phi += dt*phi_dot
        epsilon_p += dt*epsilon_p_dot
        epsilon_u += dt*epsilon_u_dot

        phis[t] = phi
        epsilon_us[t] = epsilon_u
        epsilon_ps[t] = epsilon_p

    fig = figure(figsize=(10,10))
    ax = fig.add_subplot(1,1,1)

    ax.scatter(TT,phis, s = 1, c = 'xkcd:blue', label = r'$\phi$: food size')
    ax.scatter(TT,epsilon_us, s = 1, c ='xkcd:red', label = r'$\epsilon_u$: prediction error sensory input')
    ax.scatter(TT,epsilon_ps, s = 1, c = 'xkcd:green', label = r'$\epsilon_p$: prediction error food size')

    ax.set_xlabel('Time')
    ax.legend()
    ax.set_title('Exercise 3--Neural Implementation')
    fig.savefig(join('figs',Path(__file__).stem))
    show()
