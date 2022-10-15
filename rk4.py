#!/usr/bin/env python

# Copyright (C) 2021-2022 Simon Crase

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

'''Workhorse Runge-Kutta 4th order'''

from matplotlib.pyplot import figure, show
from numpy             import array, linspace, size, zeros
from numpy.linalg      import norm

def RK4(func, y0, t):
    '''
    Runge-Kutta 4 Integrator.
    Inputs:
    func: Function name to integrate
                      this function must have two inputs namely state space
                      vector and time. For example: velocity(ssp, t)
    y0:   Initial condition, 1xd NumPy array, where d is the
                      dimension of the state space
    t:    1 x Nt NumPy array which contains instances for the solution
               to be returned.
    Outputs:
    y: d x Nt NumPy array which contains numerical solution of the
                   ODE.
    '''
    y       = zeros((size(t, 0), size(y0, 0)))
    y[0, :] = y0

    for i in range(0,size(t) - 1):
        dt       = t[i + 1] - t[i]
        k1       = dt * func(y[i], t[i])
        k2       = dt * func(y[i] + 0.5 * k1, t[i] + 0.5*dt)
        k3       = dt * func(y[i] + 0.5 * k2, t[i] + 0.5*dt)
        k4       = dt * func(y[i] + k3, t[i] + dt)
        y[i + 1] = y[i] + (k1 + 2*k2 +2*k3 +k4)/6
    return y

class KuttaMerson:
    def __init__(self,func):
        self.func = func

    def step(self,y0,h,t0):
        '''https://encyclopediaofmath.org/wiki/Kutta-Merson_method '''
        k1 = h * self.func(y0,                       t0)
        k2 = h * self.func(y0+k1/3,                  t0 + h/3)
        k3 = h * self.func(y0 + k1/6+k2/6,           t0 + h/3)
        k4 = h * self.func(y0 + k1/8 + 3*k2/8,       t0 + h/2)
        k5 = h * self.func(y0 + k1/2 -3*k3/2 + 2*k4, t0 + h)
        y1 = y0 + k1/2 -3*k3/2 + 2*k4
        y2 = y0 + k1/6         + 2*k4/3 + k5/6
        R  = 0.2 * norm(y1-y2)
        return y2,R

    def solve(self, yInit, t,ytol=1e-8,maxIter=25):
        R      = float('inf')
        m      = 0
        n      = t.shape[0]
        y      = zeros((n, yInit.shape[0]))
        y[0,:] = yInit
        for i in range(n-1):
            stepping = False
            for j in range(maxIter):
                if m>0 and R<ytol/2:
                    m-=1
                h  = (t[i+1] - t[i])/2**m
                y0 = y[i,:]
                for k in range(2**m):
                    t0 = t[i] + k*h
                    y2,R = self.step(y0,h,t0)
                    stepping = R<ytol
                    if not stepping:
                        m += 1
                        break
                    y0 = y2.copy()
                if stepping:
                    y[i+1,:] = y2
                    break
        if not stepping:
            raise Exception('oops')
        return y

if __name__ == '__main__':

    #In order to test our integration routine, we are going to define Harmonic
    #Oscillator equations in a 2D state space:
    def velocity(ssp, t):
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
        #Parameters:
        k = 1.0
        m = 1.0
        #Read inputs:
        x, v = ssp  # Read x and v from ssp vector
        #Construct velocity vector and return it:
        vel =array([v, - (k / m) * x], float)
        return vel

    #Generate an array of time points for which we will compute the solution:
    tInitial = 0
    tFinal   = 10
    Nt       = 1000  # Number of points time points in the interval tInitial, tFinal
    tArray =linspace(tInitial, tFinal, Nt)

    #Initial condition for the Harmonic oscillator:
    ssp0 =array([1.0, 0], float)

    #Compute the solution using Runge-Kutta routine:
    sspSolution = RK4(velocity, ssp0, tArray)

    #from scipy.integrate import odeint
    #sspSolution = odeint(velocity, ssp0, tArray)
    xSolution = sspSolution[:, 0]
    vSolution = sspSolution[:, 1]

    km = KuttaMerson(velocity)
    y  = km.solve(ssp0, tArray)

    fig = figure()
    ax  = fig.add_subplot(2,2,1)
    ax.plot(tArray, xSolution)
    ax.set_ylabel('x(t)')

    ax  = fig.add_subplot(2,2,2)
    ax.plot(tArray, vSolution)
    ax.set_xlabel('t (s)')
    ax.set_ylabel('v(t)')

    ax  = fig.add_subplot(2,2,3)
    ax.plot(tArray, y[:,0])
    ax  = fig.add_subplot(2,2,4)
    ax.plot( y[:,0], y[:,1])
    show()
