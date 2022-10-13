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

from argparse          import ArgumentParser
from numpy             import array, exp, linspace, mean, ones, zeros, zeros_like
from numpy.random      import default_rng
from scipy.integrate   import odeint
from matplotlib.pyplot import figure, show


class Rossler:
    def __init__(self,
                 a = 0.2,
                 b = 0.2,
                 c = 5.7):
        self.a = a
        self.b = b
        self.c = c
        self.d = 3

    def Velocity(self,t,ssp):
        x, y, z = ssp
        dxdt    = - y - z
        dydt    = x + self.a * y
        dzdt    = self.b + z * (x - self.c)
        return array([dxdt, dydt, dzdt], float)

class Lorentz:
    def __init__(self):
        self.sigma = 10.0
        self.rho   = 28.0
        self.b     = 8.0/3.0
        self.d     = 3

    def Velocity(self,t,ssp):
        x, y, z = ssp
        return array([self.sigma * (y-x),
                  self.rho*x - y - x*z,
                  x*y - self.b*z])

class Population:
    def __init__(self,oscillator,
                 stride     = 3,
                 speed    = ones(4),
                 coupling = 0.2,
                 sigma    = ones(4),
                 rng      = default_rng(42)):
        self.oscillator = oscillator
        self.stride    = stride
        self.speed     = speed
        self.coupling  = coupling
        self.sigma     = sigma
        self.rng       = rng

    def Velocity(self,t,y):

        Average  = array([mean(y[i::self.stride])  for i in range(self.oscillator.d)])
        Noise    = rng.normal(size=len(y))
        Velocity = zeros_like(y)
        for j,i in enumerate(range(0,len(y),self.stride)):
            vv = self.oscillator.Velocity(t,y[i:i+stride])
            Velocity[i:i+stride] = self.speed[j]*(vv+self.coupling*Average) + self.sigma[j] * Noise[i]
        return Velocity


if __name__ == "__main__":
    parser   = ArgumentParser(__doc__)
    parser.add_argument('oscillator',choices=['Lorentz','Rossler'])
    parser.add_argument('--tFinal', type=float, default=5)
    parser.add_argument('--Nt', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--N', type=int, default=16)
    parser.add_argument('--coupling', type=float, default=2.0)
    parser.add_argument('--show', default=False, action='store_true')
    args = parser.parse_args()

    t        = linspace(0, args.tFinal, args.Nt)

    rng      = default_rng(args.seed)
    sspInit  = rng.normal(0,1,3*args.N)
    stride    = 3
    speed    = exp(rng.normal(0,2**-3,args.N))

    sigma    = 0.00*ones(args.N)
    oscillator = Lorentz() if args.oscillator == 'Lorentz' else Rossler()
    population = Population( oscillator,
                             stride   = stride,
                             speed    = speed,
                             coupling = args.coupling,
                             sigma    = sigma,
                             rng      = rng)
    y        = odeint(population.Velocity, sspInit, t,
                        tfirst = True)

    fig = figure(figsize=(6,6))
    ax  = fig.add_subplot(1,1,1)#,projection='3d')
    m,n = y.shape
    for i in range(0,n,3):
        ax.plot(t,y[:,i])
    ax.set_title(args.oscillator)
        # ax.plot(y[:, i], y[:, i+1], y[:, i+2])
    # ax.set_xlabel('x')
    # ax.set_ylabel('y')
    # ax.set_zlabel('z')
    if args.show:
        show()
