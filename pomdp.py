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
from numpy.testing import assert_array_almost_equal

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


def update(D,B,A,
          o    = [],
          s    = [],
          log  = np.log,
          step = 1):
    def get_o(o,i):
        return o[i] if i<step else np.array([0,0])

    s1 = softmax(0.5*log(D) + 0.5*log(np.dot(B,s[1]))+log(np.dot(A,get_o(o,0))))
    s2 = softmax(0.5*log(np.dot(B,s1)) +log(np.dot(A,get_o(o,1))))

    return s1,s2



class TestSoftMax(TestCase):
    def testC(self):
        assert_array_almost_equal(np.array([[-1.1, -4.0,-2.1],
                                            [-1.1, -5.0, -3.2],
                                            [-1.1, -0.02, -0.2]]),
                                  np.log(softmax(np.array([[0,  0, 0],
                                                           [0, -1, -1],
                                                           [0,  4,  2]]))),
                                  decimal=1)

if __name__=='__main__':
    main()
