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
   Exercise 5--learn variance-- from A tutorial on the free-energy
   framework for modelling perception and learning, by Rafal Bogacz
'''

from math              import sqrt
from matplotlib.pyplot import figure, show
from matplotlib        import rc
from random            import random

rc('text', usetex=True)

class Colours:
    def __init__(self):
        self.XKCD = ['xkcd:purple','xkcd:green','xkcd:blue','xkcd:pink',
                'xkcd:brown','xkcd:red','xkcd:light blue',
                'xkcd:teal','xkcd:orange','xkcd:light green','xkcd:magenta','xkcd:yellow']

    def __iter__(self):
        self.i = 0
        return self

    def __next__(self):
        value = self.XKCD[self.i]
        self.i += 1
        return value

phi_mean  = 5
phi_sigma = 2
phi_above = 5

dt    = 0.01
MaxT  = 20
N     = 1000
LRate = 0.01

Sigma = [1]

fig = figure(figsize=(10,10))
ax  = fig.add_subplot(2,1,1)
colors = Colours()
cc     = iter(colors)

for i in range(N):
    error = [1]
    e     = [0]
    phi = phi_mean + sqrt(phi_sigma) * random()
    for j in range(int(MaxT/dt)):
        error.append(error[-1]+dt*(phi-phi_above - e[-1]))
        e.append(e[-1] + dt *(Sigma[-1] * error[-2] - e[-1]))

    Sigma.append(Sigma[-1] + LRate *(error[-1]*error[-1]-1))
    if i%100==0:
        c = next(colors)
        ax.plot(error,
                linestyle = 'dotted',
                c         = c,
                label     = 'error' if i==0 else None)
        ax.plot(e,
                linestyle = 'dashed',
                c         = c,
                label     ='e' if i==0 else None)
ax.set_title('Errors')
ax.legend()

ax  = fig.add_subplot(2,1,2)
ax.plot(Sigma)
ax.set_xlabel('Trial')
ax.set_ylabel(r'$\Sigma$')
ax.set_title(r'Evolution of $\Sigma$')
fig.suptitle('Exercise 5')
fig.savefig('feex5')
show()
