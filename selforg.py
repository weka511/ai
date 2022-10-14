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

'''
   Replication of Figure 2, Self Organization and the emergence of macroscopic behaviour,
   from Friston & Ao, Free Energy, Value, and Attractor--
   https://www.hindawi.com/journals/cmmm/2012/937860/)
'''

from argparse          import ArgumentParser
from numpy             import array, exp, linspace, mean, ones, zeros, zeros_like
from numpy.random      import default_rng
from scipy.integrate   import odeint
from matplotlib.pyplot import figure, show
from RungeKutta        import RK4

class Rossler:
    def __init__(self,
                 a = 0.2,
                 b = 0.2,
                 c = 5.7):
        self.name = 'Rossler'
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
        self.name = 'Lorentz'
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
                 speed    = ones(4),
                 coupling = 0.2,
                 sigma    = ones(4),
                 rng      = default_rng(42)):
        self.oscillator = oscillator
        self.speed      = speed
        self.coupling   = coupling
        self.sigma      = sigma
        self.rng        = rng

    def Velocity(self,y,t):                     #FIXME
        Average  = self.get_average(y)
        Noise    = rng.normal(size=len(y))
        Velocity = zeros_like(y)
        for oscillator_id,i in enumerate(range(0,len(y),self.oscillator.d)):
            OscillatorVelocity              = self.oscillator.Velocity(t,y[i:i+self.oscillator.d])
            VelocityWithCoupling            = self.speed[oscillator_id]*(OscillatorVelocity+self.coupling*Average)
            Velocity[i:i+self.oscillator.d] = VelocityWithCoupling + self.sigma[oscillator_id] * Noise[i]
        return Velocity

    def get_average(self,y):
        return array([mean(y[i::self.oscillator.d])  for i in range(self.oscillator.d)])


class OscillatorFactory:
    Oscillators = {}
    @classmethod
    def Register(cls,Oscillators):
        for oscillator in Oscillators:
            OscillatorFactory.Oscillators[oscillator.name] = oscillator
    @classmethod
    def Create(cls,name):
        return OscillatorFactory.Oscillators[name]

def parse_args():
    parser   = ArgumentParser(__doc__)
    parser.add_argument('oscillator', choices = OscillatorFactory.Oscillators.keys())
    parser.add_argument('--tFinal',   type    = float, default = 5)
    parser.add_argument('--Nt',       type    = int,   default = 1000)
    parser.add_argument('--seed',     type    = int,   default = None)
    parser.add_argument('--N',        type    = int,   default = 16)
    parser.add_argument('--coupling', type    = float, default = 2.0)
    parser.add_argument('--sigma',    type    = float, default = 2)
    parser.add_argument('--show',                      default = False, action = 'store_true')
    parser.add_argument('--burnin',   type    = int,   default = 25)
    return parser.parse_args()

if __name__ == "__main__":
    OscillatorFactory.Register([Rossler(),Lorentz()])
    args       = parse_args()
    oscillator = OscillatorFactory.Create(args.oscillator)
    t          = linspace(0, args.tFinal, args.Nt)
    rng        = default_rng(args.seed)
    population = Population(oscillator,
                            speed    = exp(rng.normal(0,2**-6,args.N)),
                            coupling = args.coupling,
                            sigma    = args.sigma * ones(args.N),
                            rng      = rng)
    y = RK4(population.Velocity,  rng.normal(30,8,oscillator.d*args.N), t)
    # y        = odeint(population.Velocity,  rng.normal(30,8,oscillator.d*args.N), t,
                        # tfirst = True)

    fig = figure(figsize=(6,6))
    for i in range(3):
        ax  = fig.add_subplot(2,2,i+1)
        ax.plot(t[args.burnin:],mean( y[args.burnin:,i::oscillator.d],axis=1),linestyle='solid')
        m,n = y.shape
        for j in range(i,n,oscillator.d):
            ax.plot(t,y[:,j],linestyle='dotted')
            ax.set_xlabel('t')
            ax.set_ylabel('xyz'[i])
    fig.suptitle(args.oscillator)
    fig.savefig(f'selforg{args.oscillator}')

    if args.show:
        show()
