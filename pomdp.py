#!/usr/bin/env python

# Copyright (C) 2023 Simon Crase

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
    POMDP example from
    Ryan Smith et al-A Step-by-Step Tutorial on Active Inference and its Application to Empirical Data
    DOI:10.31234/osf.io/b4jm6
'''

from argparse import ArgumentParser
from unittest import TestCase, main
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal

def softmax(x,axis=0):
    '''
    The softmax function converts a vector of K real numbers into a probability
    distribution of K possible outcomes.

    Parameters:
        x     The vector
        axis  Axis for summation: the default makes it agree with calculation on paage 27
    '''
    exps = np.exp(x)
    return exps/exps.sum(axis=axis)


# def update(D,B,A,
          # o    = [],
          # s    = [],
          # log  = np.log,
          # step = 1):
    # def get_o(o,i):
        # return o[i] if i<step else np.array([0,0])

    # s1 = softmax(0.5*log(D) + 0.5*log(np.dot(B,s[1]))+log(np.dot(A,get_o(o,0))))
    # s2 = softmax(0.5*log(np.dot(B,s1)) +log(np.dot(A,get_o(o,1))))

    # return s1,s2

def initialize_approximate_posteriors(priors=np.array([0.5,0.5]),T=2):
    '''
    Initialize the values of approximate posteriors for all hidden variables

    Variables:
        priors
        T        Number of time points
    '''
    Q = np.empty((T,priors.size))
    for t in range(T):
        Q[t,:] = priors
    return Q

def create_observations():
    o = np.array([[1, 0],
                  [1, 0]])
    return o

def generate_edges():
    pass

def create_messages(v):
    pass

def update_posterior():
    pass

def example1(D = np.array([0.5,0.5]),
          A = np.array([[.9,.1],
                      [.1, .9]]),
          B = np.array([[1,0],
                      [0,1]]),
          T = 2,
          N = 16):
    '''
    Example 1: Fixed observations and message passing steps

    This function carries out marginal message passing on a graph with beliefs
    about states at two time points. In this first example, both observations
    are fixed from the start (i.e., there are no ts as in full active inference
    models with sequentially presented observations) to provide the simplest
    example possible. We also highlight where each of the message passing
    steps described in the main text are carried out.

    Parameters:
        A   Likelihood mappind
        B   Transitions
        D   Priors
        T   Number of timesteps
        N   Number of iterations
    '''
    Q = np.empty((T,D.size))
    for tau in range(T):
        Q[tau,:] = D

    o = np.array([[1, 0],
                  [1, 0]])

    qs = np.empty((N + 1,D.size,T))

    for tau in range(T):
        qs[0,:,tau] =  Q[tau,:]

    for n in range(N):
        for tau in range(T):
            q = np.log(Q[tau,:])
            if tau == 0:
                lnD = np.log(D)
                lnBs = np.log(B.T.dot(Q[tau+1,:]))
            elif tau == T - 1:
                lnBs = np.log(B.dot(Q[tau-1,:]))
            lnAo = np.log(A.T.dot(o[tau,:]))
            if tau == 0:
                q = 0.5*lnD + 0.5*lnBs + lnAo
            elif tau == T-1:
                q = 0.5*lnBs + lnAo
            Q[:,tau] = softmax(q)
            qs[n+1,:,tau] =  Q[:,tau]

    return qs

class TestSoftMax(TestCase):
    def testC(self):
        '''Example on page 26'''
        assert_array_almost_equal(np.array([[-1.1, -4.0,-2.1],
                                            [-1.1, -5.0, -3.2],
                                            [-1.1, -0.02, -0.2]]),
                                  np.log(softmax(np.array([[0,  0, 0],
                                                           [0, -1, -1],
                                                           [0,  4,  2]]))),
                                  decimal=1)

    def test_initialize_approximate_posteriors(self):
        assert_array_equal(np.array([[0.5,0.5],
                                     [0.5,0.5],
                                     [0.5,0.5]]),
                           initialize_approximate_posteriors(T=3))

    def test_infer(self):
        infer()

if __name__=='__main__':
    main()
