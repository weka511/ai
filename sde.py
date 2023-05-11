#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (C) 2022-2023 Simon Crase  simon@greenweaves.nz

# This is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This software is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with GNU Emacs.  If not, see <http://www.gnu.org/licenses/>.


'''Euler–Maruyama method, based on https://en.wikipedia.org/wiki/Euler%E2%80%93Maruyama_method '''

from matplotlib.pyplot import figure, show
import numpy           as np
from numpy.random      import default_rng

class Wiener:
    '''This class represents a Wiener process'''
    def __init__(self,
                 d = 1,
                 sigma = 1,
                 rng = default_rng(None)):
        self.rng = rng
        self.d = d
        self.sigma = sigma

    def dW(self,dt):
        return self.rng.normal(size = self.d,
                               loc  = 0.0,
                               scale = np.sqrt(self.sigma*dt))

class EulerMaruyama:
    '''
    Solve SDE using Euler-Maruyama method
    See https://en.wikipedia.org/wiki/Euler%E2%80%93Maruyama_method
    '''

    def solve(self,a, y0, t,
              b  = lambda y,t:y, #Magnitude of Gaussian proportional to y
              wiener = Wiener()):
        d = np.size(y0, 0)
        y = np.zeros((np.size(t, 0), d))
        y[0, :] = y0
        for i in range(0, np.size(t) - 1):
            dt = t[i + 1] - t[i]
            y[i + 1] = y[i] + a(y[i], t[i])*dt + b(y[i],t[i])*wiener.dW(dt)

        return y

if __name__=='__main__':
    # Test code from https://en.wikipedia.org/wiki/Euler%E2%80%93Maruyama_method
    class Model:
        '''
        Stochastic model constants.
        '''
        THETA = 0.7
        MU    = 1.5
        SIGMA = 0.06

    def mu(y: float, _t: float) -> float:
        '''
        Implement the Ornstein–Uhlenbeck mu.
        '''
        return Model.THETA * (Model.MU - y)

    def sigma(_y: float, _t: float) -> float:
        '''
        Implement the Ornstein–Uhlenbeck sigma.
        '''
        return Model.SIGMA

    n = 5
    T_INIT = 3
    T_END = 7
    N = 1000  # Compute 1000 grid points
    DT = float(T_END - T_INIT) / N
    TS = np.arange(T_INIT, T_END + DT, DT)
    solver = EulerMaruyama()
    fig = figure()
    ax = fig.add_subplot(1,1,1)

    for _ in range(n):
        y = solver.solve(mu, [0], TS, b=sigma)
        ax.plot(TS,y[:,0])
    ax.set_title(f'Euler-Maruyama repeated {n} times')
    show()
