#!/usr/bin/env python

# Copyright (C) 2020-2022 Greenweaves Software Limited

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

from matplotlib.pyplot import figure, show
from matplotlib        import rc
rc('text', usetex=True)

def g(v):
    return v**2

def g_prime(v):
    return 2*v

v_p        = 3   # Mean of prior for food size
Sigma_p    = 1   # Variance of prior
Sigma_u    = 1   # Variance of sensory noise
u          = 2   # Observed light intensity

phi        = v_p  # Estimate for food size
epsilon_p  = 0    # prediction error food size
epsilon_u  = 0    # prediction error sensory input
dt         = 0.01

 # Keep track of time, and of estimates for food size and prediction errors
phis       = [phi]
epsilon_us = [epsilon_u]
epsilon_ps = [epsilon_p]
ts         = [0]

for t in range(1,501):
    phi_dot       = epsilon_u*g_prime(phi) - epsilon_p
    epsilon_p_dot = phi - v_p    - Sigma_p *epsilon_p
    epsilon_u_dot = u -   g(phi) - Sigma_u * epsilon_u

    phi          += dt*phi_dot
    epsilon_p    += dt*epsilon_p_dot
    epsilon_u    += dt*epsilon_u_dot

    ts.append(dt*t)
    phis.append(phi)
    epsilon_us.append(epsilon_u)
    epsilon_ps.append(epsilon_p)

fig = figure(figsize=(10,10))
ax  = fig.add_subplot(1,1,1)

ax.scatter(ts,phis,
           s     = 1,
           c     = 'xkcd:blue',
           label = r'$\phi$: food size')
ax.scatter(ts,epsilon_us,
           s     = 1,
           c     ='xkcd:red',
           label = r'$\epsilon_u$: prediction error sensory input')
ax.scatter(ts,epsilon_ps,
           s     = 1,
           c     = 'xkcd:green',
           label = r'$\epsilon_p$: prediction error food size')

ax.set_xlabel('Time')
ax.legend()
ax.set_title('Exercise 3--Neural Implementation')
fig.savefig('figs/feex3')
show()
