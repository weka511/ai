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

from matplotlib.pyplot import subplot, plot, xlabel, ylabel, show
from numpy             import array, linspace, size, zeros

def RK4(velocityFunction, initialCondition, timeArray):
    """
    Runge-Kutta 4 Integrator.
    Inputs:
    VelocityFunction: Function name to integrate
                      this function must have two inputs namely state space
                      vector and time. For example: velocity(ssp, t)
    InitialCondition: Initial condition, 1xd NumPy array, where d is the
                      dimension of the state space
    TimeArray: 1 x Nt NumPy array which contains instances for the solution
               to be returned.
    Outputs:
    SolutionArray: d x Nt NumPy array which contains numerical solution of the
                   ODE.
    """
    #Generate the solution array to fill in:
    SolutionArray =zeros((size(timeArray, 0),
                             size(initialCondition, 0)))
    #Assign the initial condition to the first element:
    SolutionArray[0, :] = initialCondition

    for i in range(0,size(timeArray) - 1):
        #Read time element:
        deltat = timeArray[i + 1] - timeArray[i]
        #Runge Kutta k's:
        k1 = deltat * velocityFunction(SolutionArray[i], timeArray[i])
        k2 = deltat * velocityFunction(SolutionArray[i] + 0.5 * k1, timeArray[i] + 0.5*deltat)
        k3 = deltat * velocityFunction(SolutionArray[i] + 0.5 * k2, timeArray[i] + 0.5*deltat)
        k4 = deltat * velocityFunction(SolutionArray[i] + k3, timeArray[i] + deltat)
        #Next integration step:
        SolutionArray[i + 1] = SolutionArray[i] + (k1 + 2*k2 +2*k3 +k4)/6
    return SolutionArray

if __name__ == "__main__":

    #In order to test our integration routine, we are going to define Harmonic
    #Oscillator equations in a 2D state space:
    def velocity(ssp, t):
        """
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
        """
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

    print(xSolution[-1])



    subplot(2, 1, 1)
    plot(tArray, xSolution)
    ylabel('x(t)')

    subplot(2, 1, 2)
    plot(tArray, vSolution)
    xlabel('t (s)')
    ylabel('v(t)')

    show()
