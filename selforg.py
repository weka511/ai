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

from numpy             import array, linspace
from scipy.integrate   import odeint
from matplotlib.pyplot import figure, show

A = 0.2
B = 0.2
C = 5.7


def Velocity(t,ssp,
             a = A,
             b = B,
             c = C):
    """
    Velocity function for the Rossler flow

    Inputs:
    ssp: State space vector. dx1 NumPy array: ssp=[x, y, z]
    t: Time. Has no effect on the function, we have it as an input so that our
       ODE would be compatible for use with generic integrators from
       scipy.integrate

    Outputs:
    vel: velocity at ssp. dx1 NumPy array: vel = [dx/dt, dy/dt, dz/dt]
    """

    x, y, z = ssp
    dxdt = - y - z
    dydt = x + a * y
    dzdt = b + z * (x - c)

    return array([dxdt, dydt, dzdt], float)

if __name__ == "__main__":
    tInitial = 0  # Initial time
    tFinal   = 100  # Final time
    Nt       = 10000  # Number of time points to be used in the integration

    sspSolution = odeint(Velocity, array([1, 0.0,0], float), linspace(tInitial, tFinal, Nt),
                         args=(A,B,C),
                         tfirst=True)

    fig = figure(figsize=(6,6))
    ax  = fig.add_subplot(1,1,1,projection='3d')
    ax.plot(sspSolution[:, 0], sspSolution[:, 1], sspSolution[:, 2])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    show()
