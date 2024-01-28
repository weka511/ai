#!/usr/bin/env python

# Copyright (C) 2021-2024 Simon Crase  simon@greenweaves.nz

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

'''Solve ODE using Good Old Fashioned Euler'''

from matplotlib.pyplot import figure, show
import numpy as np

def euler(func, y0, t):
    y = np.zeros((np.size(t, 0), np.size(y0, 0)))
    y[0, :] = y0

    for i in range(0,np.size(t) - 1):
        dt = t[i + 1] - t[i]
        y[i + 1] = y[i] + dt * func( t[i], y[i])

    return y

if __name__ == '__main__':
    def velocity(t,ssp,
                 k = 1.0,
                 m = 1.0):
        '''
        State space velocity function for 1D Harmonic oscillator

        Inputs:
        ssp: State space vector
        ssp = (x, v)
        t: Time. It does not effect the function, but we have t as an imput so
           that our ODE would be compatible for use with generic integrators
           from scipy.integrate

        Outputs:
        vel: Time derivative of ssp.
        vel = ds sp/dt = (v, - (k/m) x)
        '''
        x, v = ssp
        return np.array([v, - (k / m) * x], float)

    tInitial = 0
    tFinal = 10
    Nt = 5000
    tArray = np.linspace(tInitial, tFinal, Nt)

    ssp0 =np.array([1.0, 0], float)

    sspSolution = euler(velocity, ssp0, tArray)

    xSolution = sspSolution[:, 0]
    vSolution = sspSolution[:, 1]

    fig = figure(figsize=(6,6))
    ax  = fig.add_subplot(2,2,1)
    ax.plot(tArray, xSolution)
    ax.set_ylabel('x(t)')

    ax = fig.add_subplot(2,2,2)
    ax.plot(tArray, vSolution)
    ax.set_xlabel('t (s)')
    ax.set_ylabel('v(t)')

    ax  = fig.add_subplot(2,2,3)
    ax.plot( xSolution, vSolution)
    ax.set_xlabel('x(t)')
    ax.set_ylabel('v(t)')
    show()
