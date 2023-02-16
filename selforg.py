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

from abc               import ABC, abstractmethod
from argparse          import ArgumentParser
from numpy             import array, exp, linspace, mean, ones, zeros, zeros_like
from numpy.random      import default_rng
from scipy.integrate   import odeint
from matplotlib.pyplot import figure, show
from sde               import EulerMaruyama, Wiener

class Oscillator(ABC):
     @abstractmethod
     def Velocity(self,t,ssp):
          ...

class Rossler(Oscillator):
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

class Lorentz(Oscillator):
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
     '''This class represents a coupled set of Oscillators'''
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

     def Velocity(self,y,t):
          Average  = self.get_average(y)
          Noise    = rng.normal(size=len(y))
          Velocity = zeros_like(y)
          for oscillator_id,i in enumerate(range(0,len(y),self.oscillator.d)):
               OscillatorVelocity              = self.oscillator.Velocity(t,y[i:i+self.oscillator.d])
               Velocity[i:i+self.oscillator.d] = self.speed[oscillator_id]*(OscillatorVelocity+self.coupling*Average)
          return Velocity

     def get_average(self,y):
          return array([mean(y[i::self.oscillator.d])  for i in range(self.oscillator.d)])


class OscillatorFactory:
     '''This class is responsible for instantiating the appropriate Oscillator'''
     Oscillators = {}

     @classmethod
     def Register(cls,Oscillators):
          for oscillator in Oscillators:
               OscillatorFactory.Oscillators[oscillator.name] = oscillator

     @classmethod
     def Create(cls,name):
          return OscillatorFactory.Oscillators[name]

     @classmethod
     def Available(cls):
          return OscillatorFactory.Oscillators.keys()

def parse_args():
     parser   = ArgumentParser(__doc__)
     parser.add_argument('oscillator', choices = OscillatorFactory.Available())
     parser.add_argument('--tFinal',   type    = float, default = 8)
     parser.add_argument('--Nt',       type    = int,   default = 1024)
     parser.add_argument('--seed',     type    = int,   default = None)
     parser.add_argument('--N',        type    = int,   default = 16)
     parser.add_argument('--coupling', type    = float, default = 1.0)
     parser.add_argument('--sigma',    type    = float, default = 1.0)
     parser.add_argument('--sigma0',   type    = float, default = 8.0)
     parser.add_argument('--show',                      default = False, action = 'store_true')
     parser.add_argument('--burnin',   type    = int,   default = 0)
     return parser.parse_args()

if __name__ == "__main__":
     OscillatorFactory.Register([Rossler(),Lorentz()])
     args       = parse_args()
     oscillator = OscillatorFactory.Create(args.oscillator)
     t          = linspace(0, args.tFinal, args.Nt)
     d          = 3 * args.N
     rng        = default_rng(args.seed)
     y0         = rng.normal(30, args.sigma0, oscillator.d*args.N)
     population = Population(oscillator,
                             speed    = exp(rng.normal(0,2**-6,args.N)),
                             coupling = args.coupling,
                             sigma    = args.sigma * ones(args.N),
                             rng      = rng)
     solver    = EulerMaruyama()
     y         = solver.solve(population.Velocity, y0, t,
                              b       = lambda y,t:args.sigma * y, #Magnitude of Gaussian proportional to y
                              wiener  = Wiener(rng = rng,
                                               d   = d))

     fig       = figure(figsize=(12,6))
     for i in range(3):
          ax  = fig.add_subplot(2,2,i+1)
          ax.plot(t[args.burnin:],mean( y[args.burnin:,i::oscillator.d],axis=1),linestyle='solid',linewidth=2)
          m,n = y.shape
          for j in range(i,n,oscillator.d):
               ax.plot(t[args.burnin:],y[args.burnin:,j],linestyle='dotted')
               ax.set_xlabel('t')
               ax.set_ylabel('xyz'[i])

     ax  = fig.add_subplot(2,2,4,projection='3d')
     ax.plot(mean( y[args.burnin:,0::oscillator.d],axis=1),
             mean( y[args.burnin:,1::oscillator.d],axis=1),
             mean( y[args.burnin:,2::oscillator.d],axis=1))
     ax.set_xlabel('x')
     ax.set_ylabel('y')
     ax.set_zlabel('z')
     fig.suptitle(fr'{args.oscillator}: N={args.N}, $\lambda$={args.coupling}, $\sigma=${args.sigma}, $\delta T=${args.tFinal/args.Nt}')
     fig.savefig(f'selforg{args.oscillator}')

     if args.show:
         show()
