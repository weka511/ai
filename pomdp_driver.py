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
from os.path import join
from pathlib import Path
from matplotlib.pyplot import figure,rcParams,show
import numpy as np
from pomdp import example1

def parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('action', type=int)
    parser.add_argument('--show', default=False, action='store_true')
    parser.add_argument('--figs', default='./figs', help='Path to plot files (default: %(default)s)')
    parser.add_argument('--plot', default = None)
    return parser.parse_args()

def get_plotfile_name(args):
    if args.plot == None:
        return f'{Path(__file__).stem}_{args.action}'
    else:
        return args.plot

if __name__ == '__main__':
    args = parse_args()
    rcParams['text.usetex'] = True
    match args.action:
        case 1:
            qs = example1()
            fig = figure()
            ax = fig.add_subplot(1,1,1)
            _,m,T = qs.shape
            for i in range(m):
                for tau in range(T):
                    ax.plot(qs[:,i,tau],label=fr'{i}, $\tau=${tau}')
            ax.legend()
            ax.set_title('Example 1: Fixed observations and message passing steps')
            fig.savefig(join(args.figs,get_plotfile_name(args)))

    if args.show:
        show()


# def display(name,matrices):
    # print (f'{name}')
    # for matrix in matrices:
        # print (f'{matrix}')

# # ---------------------------------------------------------------------
# # State factors

# # 0. left-better_context/right_better_context
# # 1. pre_choice/asking_for_hint/choosing_left_machine/choosing_right_machine

# # ---------------------------------------------------------------------

# # Priors over initial states

# D = [
    # np.array([0.5, 0.5]),       # left-better_context/right_better_context
    # np.array([1.0, 0, 0, 0])    # pre_choice/asking_for_hint/choosing_left_machine/choosing_right_machine
# ]

# # ---------------------------------------------------------------------

# # Observations

# A = [
    # np.array([
        # [[1,1],
         # [0,0],
         # [0,0]],
        # [[0,0],
         # [1,0],
         # [0,1]],
        # [[1,1],
         # [0,0],
         # [0,0]],
        # [[1,1],
         # [0,0],
         # [0,0]]
        # ]),
    # np.array([
        # [[1,1],
         # [0,0],
         # [0,0]],
        # [[1,1],
         # [0,0],
         # [0,0]],
        # [[0,0],
         # [0.2,0.0],
         # [0.8,0.2]],
        # [[0,0],
         # [0.8,0.2],
         # [0.2,0.8]]
        # ]),
    # np.array ([
        # [[1,1],
         # [0,0],
         # [0,0],
         # [0,0]],
        # [[0,0],
         # [1,1],
         # [0,0],
         # [0,0]],
        # [[0,0],
         # [0,0],
         # [1,1],
         # [0,0]],
        # [[0,0],
         # [0,0],
         # [0,0],
         # [1,1]]
    # ])
# ]

# # ---------------------------------------------------------------------

# # State transition matrices

# B = [
    # np.array([[1,0],
              # [0,1]]),
    # np.array([[[1,1,1,1],
               # [0,0,0,0],
               # [0,0,0,0],
               # [0,0,0,0]],
              # [[0,0,0,0],
               # [1,1,1,1],
               # [0,0,0,0],
               # [0,0,0,0]],
              # [[0,0,0,0],
               # [0,0,0,0],
               # [1,1,1,1],
               # [0,0,0,0]],
              # [[0,0,0,0],
               # [0,0,0,0],
               # [0,0,0,0],
               # [1,1,1,1]]
            # ])]

# # ---------------------------------------------------------------------

# # preferences for outcomes

# C = [
    # np.zeros((3,3)),
    # np.array([[0,  0, 0],
              # [0, -1, -1],
              # [0,  4,  2]]),
    # np.zeros((4,3))
# ]

# # ---------------------------------------------------------------------

# # Shallow policies

# U = [
    # np.array([[1,1,1,1],
              # [1,2,3,4]])
# ]

# # ---------------------------------------------------------------------

# # Deep policies

# V = [
    # np.array([[1, 1, 1, 1, 1],
              # [1, 1, 1, 1, 1]]),
    # np.array([[1, 2, 2, 3, 4],
              # [1, 3, 4, 1, 1]])
# ]


# # ---------------------------------------------------------------------

# # Habits

# E = np.ones((5))/5
# display ('D', D)
# display ('A', A)
# display ('B', B)
# display ('C', C)
# display ('U', U)
# display ('V', V)
# display ('E', E)
