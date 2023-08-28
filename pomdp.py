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
from sys import float_info
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
    pass

def generate_edges():
    pass

def create_messages(v):
    pass

def update_posterior():
    pass

def pass_messages(A,B,C,D,E,U,V,
                  max_eps=0.1,
                  T=2,
                  N=16):
    q = initialize_approximate_posteriors(priors=D,T=T)
    o = create_observations()
    for v in generate_edges():
        eps = float_info.max
        while eps > max_eps:
            mus = create_messages(v)
            # pass messages
            q = update_posterior()

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

if __name__=='__main__':
    main()
