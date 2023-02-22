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

'''Message passing example: ported from https://github.com/rssmith33/Active-Inference-Tutorial-Scripts/blob/main/Message_passing_example.m'''


# Standard library imports.

from argparse import ArgumentParser

# Related third party imports.

from matplotlib.pyplot import figure, show
import numpy as np

# Local application/library specific imports.

from pomdp import softmax

if __name__=='__main__':
    D = np.array([.5, .5]) # Priors

    A = np.array([ #likelihood mapping
                   [.9, .1],
                   [.1, .9]])

    B = np.array([  # transitions
                    [1, 0],
                    [0, 1]])

    T = 2 #number of timesteps

    N = 16 # number of iterations of message passing

    Qs = np.array([[.5, .5] for _ in range(T)]) #  initialize posterior (Step 1)

    o = [[1, 0] for _ in range(T)] # fix observations (Step 2)

    qs = np.zeros((N,2,T))  # FIXME

    # iterate a set number of times (alternatively, until convergence) (Step 8)
    for i in range(N):
        for tau in range(T): # For each edge (hidden state) (Step 7)
            q = np.log(Qs[:,tau])
            #compute messages sent by D and B (Steps 4) using the posterior
            # computed in Step 6B
            if tau == 0:
                lnD  = np.log(D);                # Message 1
                lnBs = np.log(np.dot(B,Qs[:,tau+1]))  # Message 2
            elif tau == T-1: # last time point
                lnBs = np.log(np.dot(B,Qs[:,tau-1]))  # Message 1  FIXME

            lnAo = np.log(np.dot(A,o[tau])) #  likelihood (Message 3)

            # Steps 5-6 (Pass messages and update the posterior)
            #   % Since all terms are in log space, this is addition instead of
            #   % multiplication. This corresponds to  equation 16 in the main
            #   % text (within the softmax)
            if tau == 0:
                q = .5*lnD + .5*lnBs + lnAo
            elif tau == T-1:
                q = .5*lnBs + lnAo

            Qs[:,tau] = softmax(q)
            qs[i,:,tau] = Qs[:,tau]


    fig = figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(qs[:,:,0], label='q0')
    ax.plot(qs[:,:,1], label='q1')
    ax.set_title('Example 1: Approximate posteriors (1 per edge per time point)')
    ax.set_xlabel('Message passing iterations')
    ax.legend()
    show()

