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

v_p        = 3
Sigma_p    = 1
Sigma_u    = 1
u          = 2
phi        = v_p
epsilon_p  = 0
epsilon_u  = 0
dt         = 0.01

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
           label = r'$\phi$')
ax.scatter(ts,epsilon_us,
           s     = 1,
           c     ='xkcd:red',
           label = r'$\epsilon_u$')
ax.scatter(ts,epsilon_ps,
           s     = 1,
           c     = 'xkcd:green',
           label = r'$\epsilon_p$')

ax.set_xlabel('Time')
ax.legend()
ax.set_title('Exercise 3')
fig.savefig('figs/feex3')
show()
