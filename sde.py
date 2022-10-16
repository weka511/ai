#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (C) 2022 Simon Crase

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

# Copyright <c) Wikipedia contributors 2015-2022
# Creative Commons Attribution-ShareAlike 3.0 License
# https://creativecommons.org/licenses/by-sa/3.0/
# Wikipedia contributors, "Euler–Maruyama method," Wikipedia, The Free Encyclopedia, https://en.wikipedia.org/w/index.php?title=Euler%E2%80%93Maruyama_method&oldid=1104040790 (accessed October 15, 2022).
#
# I have made typographical changes to confirm to the style of my other code

'''Euler–Maruyama method, based on https://en.wikipedia.org/w/index.php?title=Euler%E2%80%93Maruyama_method&oldid=1104040790 '''

from matplotlib.pyplot import figure, show
from numpy             import arange, size, sqrt, zeros
from numpy.random      import normal

class EulerMaruyama:
    def __init__(self):
        pass

    def dW(step0,dt,size=1,p=[]):
        sigma = p[0] if len(p)>0 else 1
        return normal(size=1,loc=0.0, scale=sqrt(sigma*dt))

    def solve(self,a,b, y0, t,p=[]):
        d       = size(y0, 0)
        y       = zeros((size(t, 0), d))
        y[0, :] = y0

        for i in range(0,size(t) - 1):
            dt       = t[i + 1] - t[i]
            y[i + 1] = y[i] + a(y[i], t[i]) * dt + b(y[i], t[i]) * self.dW(dt,size=d,p=[])

        return t,y



if __name__=='__main__':
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

    T_INIT = 3
    T_END  = 7
    N      = 1000  # Compute 1000 grid points
    DT     = float(T_END - T_INIT) / N
    TS = arange(T_INIT, T_END + DT, DT)
    euler_maruyama = EulerMaruyama()
    fig = figure()
    ax  = fig.add_subplot(1,1,1)

    for _ in range(5):
        t,y = euler_maruyama.solve(mu,sigma, [0], TS)
        ax.plot(t,y[:,0])

    show()
