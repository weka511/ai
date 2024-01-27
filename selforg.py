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

from abc import ABC, abstractmethod
from argparse import ArgumentParser
import numpy as np
from scipy.integrate import odeint
from matplotlib.pyplot import figure, show
from sde import EulerMaruyama, Wiener

class Oscillator(ABC):
     def __init__(self,name,d=3):
          self.name = name
          self.d = d

     @abstractmethod
     def Velocity(self,t,ssp):
          ...

class Huyghens(Oscillator):
     def __init__(self,omega2=1):
          super().__init__('Huyghens',d=2)
          assert omega2>0
          self.omega2 = omega2

     def Velocity(self,t,ssp):
          x, y = ssp
          return np.array(y,-self.omega2*np.sin(x))

class Rossler(Oscillator):

     def __init__(self,
                  a = 0.2,
                  b = 0.2,
                  c = 5.7):
          super().__init__('Rössler')
          self.a = a
          self.b = b
          self.c = c

     def Velocity(self,t,ssp):
          x, y, z = ssp
          dxdt = - y - z
          dydt = x + self.a * y
          dzdt = self.b + z * (x - self.c)
          return np.array([dxdt, dydt, dzdt], float)

class Lorentz(Oscillator):
     def __init__(self):
          super().__init__('Lorentz')
          self.sigma = 10.0
          self.rho   = 28.0
          self.b     = 8.0/3.0

     def Velocity(self,t,ssp):
          x, y, z = ssp
          return np.array([self.sigma * (y-x),
                        self.rho*x - y - x*z,
                        x*y - self.b*z])

class Population:
     '''This class represents a coupled set of Oscillators'''
     def __init__(self,oscillator,
                  speed = np.ones(4),
                  coupling = 0.2,
                  rng = np.random.default_rng(42)):
          self.oscillator = oscillator
          self.speed = speed
          self.coupling = coupling
          self.rng = rng

     def Velocity(self,y,t):
          Average = self.get_average(y)
          Noise = rng.normal(size=len(y))
          Velocity = np.zeros_like(y)
          for oscillator_id,i in enumerate(range(0,len(y),self.oscillator.d)):
               OscillatorVelocity = self.oscillator.Velocity(t,y[i:i + self.oscillator.d])
               Velocity[i:i+self.oscillator.d] = self.speed[oscillator_id] * (OscillatorVelocity + self.coupling*Average)
          return Velocity

     def get_average(self,y):
          return np.array([np.mean(y[i::self.oscillator.d])  for i in range(self.oscillator.d)])


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

def parse_args(tFinal = 8,
               Nt = 1024,
               seed = None,
               N = 16,
               coupling = 1.0,
               sigma = 1.0,
               sigma0 = 8.0,
               burnin = 0,
               loc = 30.0):
     parser = ArgumentParser(__doc__)
     parser.add_argument('oscillator', choices = OscillatorFactory.Available())
     parser.add_argument('--tFinal', type = float, default = tFinal, help= f'Final time for integration [{tFinal}]')
     parser.add_argument('--Nt', type = int,   default = Nt, help= f'Number of time steps [{Nt}]')
     parser.add_argument('--seed', type = int,   default = seed, help= f'Seed for random number generation [{seed}]')
     parser.add_argument('--N', type = int,   default = N, help= f'Number of oscillator s[{N}]')
     parser.add_argument('--coupling', type = float, default = coupling, help= f'Coupling coefficient for oscillators [{coupling}]')
     parser.add_argument('--sigma', type = float, default = sigma, help= f'Standard deviation for noise [{sigma}]')
     parser.add_argument('--sigma0', type = float, default = sigma0, help= f'Used to disperse starting points [{sigma0}]')
     parser.add_argument('--show', default = False, action = 'store_true', help= 'Set if plots are to be displayed')
     parser.add_argument('--burnin', type = int, default = burnin, help= f'Burning-in time (omitted from plots)[{burnin}]')
     parser.add_argument('--loc', type=float, default=loc, help= f'Mean for initial positions [{loc}]')
     return parser.parse_args()

if __name__ == "__main__":
     OscillatorFactory.Register([Rossler(),
                                 Lorentz(),
                                 Huyghens()])
     args = parse_args()
     oscillator = OscillatorFactory.Create(args.oscillator)
     t = np.linspace(0, args.tFinal, args.Nt)
     d = oscillator.d * args.N
     rng = np.random.default_rng(args.seed)
     y0 = rng.normal(loc = args.loc,
                     scale = args.sigma0,
                     size = oscillator.d*args.N)
     population = Population(oscillator,
                             speed = np.exp(rng.normal(0,2**-6,args.N)),
                             coupling = args.coupling,
                             rng = rng)
     solver = EulerMaruyama()
     y = solver.solve(population.Velocity, y0, t,
                      b = lambda y,t:args.sigma * y, #Magnitude of Gaussian proportional to y
                      wiener = Wiener(rng = rng, d = d))

     fig = figure(figsize=(12,6))
     for i in range(3):
          ax = fig.add_subplot(2,2,i+1)
          ax.plot(t[args.burnin:],np.mean(y[args.burnin:,i::oscillator.d],axis=1),
                  linestyle = 'solid',
                  linewidth = 2)
          m,n = y.shape
          for j in range(i,n,oscillator.d):
               ax.plot(t[args.burnin:],y[args.burnin:,j],
                       linestyle = 'dotted')
               ax.set_xlabel('t')
               ax.set_ylabel('xyz'[i])

     ax = fig.add_subplot(2,2,4,projection='3d')
     ax.plot(np.mean(y[args.burnin:,0::oscillator.d],axis=1),
             np.mean(y[args.burnin:,1::oscillator.d],axis=1),
             np.mean(y[args.burnin:,2::oscillator.d],axis=1))
     ax.set_xlabel('x')
     ax.set_ylabel('y')
     ax.set_zlabel('z')
     fig.suptitle(fr'{args.oscillator}: N={args.N}, $\lambda$={args.coupling}, $\sigma=${args.sigma}, $\delta T=${args.tFinal/args.Nt}')
     fig.savefig(f'figs/selforg{args.oscillator}')

     if args.show:
          show()
