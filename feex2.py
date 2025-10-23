#!/usr/bin/env python

# Copyright (C) 2020-25 Greenweaves Software Limited

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
   Exercise 2, posterior probabilities, from A tutorial on the free-energy
   framework for modelling perception and learning, by Rafal Bogacz.

   Find the most likely size for the food item
'''

from os.path import join
from pathlib import Path
from matplotlib.pyplot import figure, show
from matplotlib import rc
import numpy as np

vp = 3    # Mean of prior for food size
Sigma_p = 1    # Variance of prior
Sigma_u = 1    # Variance of sensory noise
u = 2   # Observed light intensity


def generate_phi(phi0, N = 500, g = lambda v:v**2, g_prime = lambda v: 2*v, dt = 0.01):
    '''
    Generate successive estimates for size of food item

    Parameters:
        phi0        Initial estimate for size
        N = 500
        g = lambda v:v**2
        g_prime = lambda v: 2*v
        dt
    '''
    phi = phi0
    yield phi
    for i in range(N):
        df = (vp - phi)/Sigma_p + (u - g(phi))*g_prime(phi)/Sigma_u
        phi += dt*df
        yield phi

if __name__ == '__main__':
    rc('text', usetex=True)
    T0 = 0
    T1 = 5
    N = 500
    Ts = np.linspace(T0,T1,num=N+1)
    Phis = np.array(list(generate_phi(vp,N=N,dt=(T1-T0)/N)))

    fig = figure(figsize=(10,10))
    ax  = fig.add_subplot(1,1,1)
    ax.scatter(Ts,Phis, s = 1, c = 'xkcd:blue', label = 'Most likely size of food item')
    ax.set_title('Exercise 2')
    ax.set_ylim(0,3)
    ax.set_xlabel('t')
    ax.set_ylabel(r'$\phi$')
    ax.legend()
    fig.savefig(join('figs',Path(__file__).stem))
    show()
