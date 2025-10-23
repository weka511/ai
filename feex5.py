#!/usr/bin/env python

# Copyright (C) 2020-2023 Greenweaves Software Limited

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

from os.path import join
from pathlib import Path
import numpy as np
from matplotlib.pyplot import figure, show
from matplotlib import rc

class Colours:
    def __init__(self):
        self.XKCD = [
            'xkcd:purple','xkcd:green','xkcd:blue','xkcd:pink',
            'xkcd:brown','xkcd:red','xkcd:light blue','xkcd:teal',
            'xkcd:orange','xkcd:light green','xkcd:magenta','xkcd:yellow'
        ]

    def __iter__(self):
        self.i = 0
        return self

    def __next__(self):
        value = self.XKCD[self.i]
        self.i += 1
        return value

if __name__=='__main__':
    rc('text', usetex=True)
    rng = np.random.default_rng()
    phi_mean = 5
    phi_sigma = 2
    phi_above = 5

    dt = 0.01
    MaxT = 20
    N = 1000
    LRate = 0.01
    Sigma = np.zeros((N+1))
    Sigma[0] = 1

    fig = figure(figsize=(10,10))
    ax  = fig.add_subplot(2,1,1)
    colours = Colours()
    colour_iterator = iter(colours)
    m = int(MaxT/dt)
    e = np.zeros((m+1))   # interneuron
    error = np.zeros((m+1))  # prediction error
    for i in range(N):
        phi = phi_mean + np.sqrt(phi_sigma)*rng.uniform()
        for j in range(m):
            error[j+1] = error[j] + dt*(phi-phi_above - e[j])
            e[j+1] = e[j] + dt *(Sigma[i] * error[j] - e[j])

        Sigma[i+1] = Sigma[i] + LRate *( error[-1]*e[-1]- 1)

        if i % 100==0:
            colour = next(colour_iterator)
            ax.plot(error, linestyle = 'dotted', c = colour, label = 'prediction error' if i==0 else None)
            ax.plot(e, linestyle = 'dashed',  c = colour,  label ='interneuron' if i==0 else None)

    ax.set_title(f'Errors over {N} runs')
    ax.legend()

    ax  = fig.add_subplot(2,1,2)
    ax.plot(Sigma)
    ax.set_xlabel('Trial')
    ax.set_ylabel(r'$\Sigma$')
    ax.set_title(r'Evolution of $\Sigma$')

    fig.suptitle('Exercise 5: learn variance')
    fig.savefig(join('figs',Path(__file__).stem))
    show()
