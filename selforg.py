#!/usr/bin/env python

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

'''Self Organization and the emergence of macroscopic behaviour'''

from numpy             import array, linspace, mean, ones, zeros, zeros_like
from numpy.random      import default_rng
from scipy.integrate   import odeint
from matplotlib.pyplot import figure, show

A = 0.2
B = 0.2
C = 5.7


def Velocity(t,y,
             a        = A,
             b        = B,
             c        = C,
             step     = 3,
             speed    = ones(4),
             coupling = 0.2,
             sigma    = ones(4),
             rng      = default_rng(42)):
    '''
    Velocity function for the Rossler flow

    Inputs:
    ssp: State space vector. dx1 NumPy array: ssp=[x, y, z]
    t: Time. Has no effect on the function, we have it as an input so that our
       ODE would be compatible for use with generic integrators from
       scipy.integrate

    Outputs:
    vel: velocity at ssp. dx1 NumPy array: vel = [dx/dt, dy/dt, dz/dt]
    '''


    v    = zeros_like(y)
    average = zeros(3)
    for i in range(step):
        average[i] = mean(y[i::step])
    noise = rng.normal(size=len(y))
    for j,i in enumerate(range(0,len(y),step)):
        v[i]   = speed[j]*(- y[i+1] - y[i+2] +coupling*average[0]) + sigma[j] * noise[i]
        v[i+1] = speed[j]*(y[i] + a * y[i+1]+coupling*average[1]) + sigma[j] * noise[i+1]
        v[i+2] = speed[j]*(b + y[i+2] * (y[i] - c)+coupling*average[2]) + sigma[j] * noise[i+2]

    return v


if __name__ == "__main__":
    tInitial = 0
    tFinal   = 10
    Nt       = 1000
    seed     = 42
    N        = 16
    rng      = default_rng(seed)
    sspInit  = rng.normal(0,1,3*N)
    step     = 3
    speed    = ones(N)
    coupling = 2.0
    sigma    = 0.001*ones(N)
    y        = odeint(Velocity, sspInit, linspace(tInitial, tFinal, Nt),
                      args   = (A,B,C,step,speed,coupling,sigma,rng),
                      tfirst = True)

    fig = figure(figsize=(6,6))
    ax  = fig.add_subplot(1,1,1,projection='3d')
    m,n = y.shape
    for i in range(0,n,3):
        ax.plot(y[:, i], y[:, i+1], y[:, i+2])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    show()
