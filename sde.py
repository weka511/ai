#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright <c) Wikipedia contributors 2015-2022
# Creative Commons Attribution-ShareAlike 3.0 License
# https://creativecommons.org/licenses/by-sa/3.0/
# Wikipedia contributors, "Euler–Maruyama method," Wikipedia, The Free Encyclopedia, https://en.wikipedia.org/w/index.php?title=Euler%E2%80%93Maruyama_method&oldid=1104040790 (accessed October 15, 2022).
#
# I have made typographical changes to confirm to the style of my other code

'''Euler–Maruyama method'''

from matplotlib.pyplot import figure, show
from numpy             import arange, sqrt, zeros
from numpy.random      import normal


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

def dW(delta_t: float) -> float:
    '''
    Sample a random number at each call.
    '''
    return normal(loc=0.0, scale=sqrt(delta_t))

def run_simulation():
    '''
    Return the result of one full simulation.
    '''
    T_INIT = 3
    T_END  = 7
    N      = 1000  # Compute 1000 grid points
    DT     = float(T_END - T_INIT) / N
    TS = arange(T_INIT, T_END + DT, DT)

    Y_INIT = 0

    ys = zeros(N + 1)
    ys[0] = Y_INIT
    for i in range(1, TS.size):
        t = T_INIT + (i - 1) * DT
        y = ys[i - 1]
        ys[i] = y + mu(y, t) * DT + sigma(y, t) * dW(DT)

    return TS, ys

def plot_simulations(num_sims: int) -> None:
    '''
    Plot several simulations in one image.
    '''
    fig = figure()
    ax  = fig.add_subplot(1,1,1)

    for _ in range(num_sims):
        ax.plot(*run_simulation())

    ax.set_xlabel('time')
    ax.set_ylabel('y')


if __name__=='__main__':
    NUM_SIMS = 5
    plot_simulations(NUM_SIMS)
    show()
