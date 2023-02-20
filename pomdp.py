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
    https://www.researchgate.net/publication/348153427_A_Step-by-Step_Tutorial_on_Active_Inference_and_its_Application_to_Empirical_Data
    and https://github.com/rssmith33/Active-Inference-Tutorial-Scripts/blob/main/Step_by_Step_AI_Guide.m
'''

from argparse import ArgumentParser

import numpy as np

def softmax(x):
    exps = np.exp(x)
    return exps/exps.sum()



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

# ---------------------------------------------------------------------
# State factors

# 0. left-better_context/right_better_context
# 1. pre_choice/asking_for_hint/choosing_left_machine/choosing_right_machine

# ---------------------------------------------------------------------

# Priors over initial states

D = [
    np.array([0.5, 0.5]),       # left-better_context/right_better_context
    np.array([1.0, 0, 0, 0])    # pre_choice/asking_for_hint/choosing_left_machine/choosing_right_machine
]

# ---------------------------------------------------------------------

# Observations

A = [
    np.array([
        [[1,1],
         [0,0],
         [0,0]],
        [[0,0],
         [1,0],
         [0,1]],
        [[1,1],
         [0,0],
         [0,0]],
        [[1,1],
         [0,0],
         [0,0]]
        ]),
    np.array([
        [[1,1],
         [0,0],
         [0,0]],
        [[1,1],
         [0,0],
         [0,0]],
        [[0,0],
         [0.2,0.0],
         [0.8,0.2]],
        [[0,0],
         [0.8,0.2],
         [0.2,0.8]]
        ]),
    np.array ([
        [[1,1],
         [0,0],
         [0,0],
         [0,0]],
        [[0,0],
         [1,1],
         [0,0],
         [0,0]],
        [[0,0],
         [0,0],
         [1,1],
         [0,0]],
        [[0,0],
         [0,0],
         [0,0],
         [1,1]]
    ])
]

# ---------------------------------------------------------------------

# State transition matrices

B = [np.array([[1,0],
            [0,1]]),
     np.array([[[1,1,1,1],
            [0,0,0,0],
            [0,0,0,0],
            [0,0,0,0]],
            [[0,0,0,0],
             [1,1,1,1],
             [0,0,0,0],
             [0,0,0,0]],
            [[0,0,0,0],
             [0,0,0,0],
             [1,1,1,1],
             [0,0,0,0]],
            [[0,0,0,0],
             [0,0,0,0],
             [0,0,0,0],
             [1,1,1,1]]
            ])]

# ---------------------------------------------------------------------

# preferences for outcomes

C = [np.zeros((3,3)),
     np.zeros((4,3)),
     np.array([[0,  0, 0],
            [0, -1, -1],
            [0, 4,  2]])]



# ---------------------------------------------------------------------

# Allowable policies

U = []

# ---------------------------------------------------------------------

# Deep policies

V = []


# ---------------------------------------------------------------------

# Habits

E = []
if __name__=='__main__':
    print ('D')
    print (D[0])
    print (D[1])
    print ('A')
    print (A[0])
    print (A[1])
    print (A[2])
    print ('B')
    print(B[0])
    print(B[1])
    print ('C')
    print(C[0])
    print(C[1])
    print(C[2])
