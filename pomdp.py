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
    https://psyarxiv.com/b4jm6/
    and https://github.com/rssmith33/Active-Inference-Tutorial-Scripts/blob/main/Step_by_Step_AI_Guide.m
'''

from argparse import ArgumentParser
from numpy    import array, zeros

# ---------------------------------------------------------------------
# State factors

# 0. left-better_context/right_better context
# 1. pre_choice/asking_for_hint/choosing_left_machine/choosing_right_machine

# ---------------------------------------------------------------------

# Priors over initial states

D = [
    array([0.5, 0.5]),
    array([1.0, 0, 0, 0])
]

# ---------------------------------------------------------------------

# Observations

A = [
    array([
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
    array([
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
    array ([
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

B = [array([[1,0],
            [0,1]]),
     array([[[1,1,1,1],
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

C = [zeros((3,3)),
     zeros((4,3)),
     array([[0,  0, 0],
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