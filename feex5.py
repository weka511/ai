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
    '''
    Provide a selection of colours from XKCD model
    '''

    def __init__(self):
        self.XKCD = [
            'xkcd:purple', 'xkcd:green', 'xkcd:blue', 'xkcd:pink',
            'xkcd:brown', 'xkcd:red', 'xkcd:light blue', 'xkcd:teal',
            'xkcd:orange', 'xkcd:light green', 'xkcd:magenta', 'xkcd:yellow'
        ]

    def __iter__(self):
        self.i = 0
        return self

    def __next__(self):
        if self.i >= len(self.XKCD):
            raise StopIteration()
        value = self.XKCD[self.i]
        self.i += 1
        return value


if __name__ == '__main__':
    rc('text', usetex=True)
    rng = np.random.default_rng()
    phi_mean = 5
    phi_sigma = 2
    phi_above = 5

    dt = 0.01
    MaxT = 20
    N = 1000
    LRate = 0.01
    m = int(MaxT / dt)
    e = np.zeros((m + 1))   # interneuron
    error = np.zeros((m + 1))  # prediction error
    Sigma = np.ones((N + 1))

    fig = figure(figsize=(10, 10))
    ax = fig.add_subplot(2, 1, 1)
    colours = Colours()
    colour_iterator = iter(colours)

    for trial in range(N):
        phi = phi_mean + np.sqrt(phi_sigma) * rng.uniform()
        error[0] = 0
        e[0] = 0
        for i in range(m):
            error[i + 1] = error[i] + dt * (phi - phi_above - e[i])
            e[i + 1] = e[i] + dt * (Sigma[trial] * error[i] - e[i])

        Sigma[trial + 1] = Sigma[trial] + LRate * (error[-1] * e[-1] - 1)

        if trial % 100 == 0:
            colour = next(colour_iterator)
            ax.plot(error, linestyle='dotted', c=colour, label='prediction error' if trial == 0 else None)
            ax.plot(e, linestyle='dashed', c=colour, label='interneuron' if trial == 0 else None)

    ax.set_title(f'Errors over {N} runs')
    ax.legend()

    ax = fig.add_subplot(2, 1, 2)
    ax.plot(Sigma)
    ax.set_xlabel('Trial')
    ax.set_ylabel(r'$\Sigma$')
    ax.set_title(r'Evolution of $\Sigma$')

    fig.suptitle('Exercise 5: learn variance')
    fig.savefig(join('figs', Path(__file__).stem))
    show()
