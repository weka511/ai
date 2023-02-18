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

'''Solve problem of Example 2 from Smith et al'''

import numpy as np

def softmax(x):
    exps = np.exp(x)
    return exps/exps.sum()



def step1(D,B,A,log=np.log):
    return array([softmax(0.5*log(D) + 0.5*log(np.dot(B,s2))+log(np.dot(A,o1))),
                  softmax(0.5*log(D) +log(np.dot(A,np.array([0,0]))))])

D = np.array([0.75, 0.25])

A = np.array([[0.8, 0.2],
           [0.2, 0.8]])

B = np.array([[0,1],
          [0,1]])

o1 = np.array([1,0])

o2 = np.array([0,1])

s1 = np.array([0.5, 0.5])

s2 = np.array([0.5, 0.5])

print (step1(D,B,A,log=lambda x:np.log(x+0.01)))

