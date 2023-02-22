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
   Message passing example: ported from
   https://github.com/rssmith33/Active-Inference-Tutorial-Scripts/blob/main/Message_passing_example.m
'''

from argparse import ArgumentParser
from os.path  import join
from pathlib  import Path

from matplotlib.pyplot import figure, show
import numpy as np

from pomdp import softmax

if __name__=='__main__':
    parser = ArgumentParser(__doc__)
    parser.add_argument('--show', default=False, action='store_true')
    parser.add_argument('--figs', default='./figs')
    args = parser.parse_args()

    # Example 1: Fixed observations and message passing steps

    # This section carries out marginal message passing on a graph with beliefs
    # about states at two time points. In this first example, both observations
    # are fixed from the start (i.e., there are no ts as in full active inference
    # models with sequentially presented observations) to provide the simplest
    # example possible. We also highlight where each of the message passing
    # steps described in the main text are carried out.

    # Note that some steps (7 and 8) appear out of order when they involve loops that
    # repeat earlier steps

    D  = np.array([.5, .5])     # Priors

    A  = np.array([             #likelihood mapping
                   [.9, .1],
                   [.1, .9]])

    B  = np.array([             # transitions
                    [1, 0],
                    [0, 1]])

    T  = 2                     # number of timesteps
    N  = 16                    # number of iterations of message passing

    Qs = np.array([[.5, .5] for _ in range(T)]) #  initialize posterior (Step 1)

    o  = [[1, 0] for _ in range(T)] # fix observations (Step 2)

    qs = np.zeros((N+1,D.shape[0],T))  # Used for plotting, not for calculations
    for tau in range(T):
        qs[0,:,tau] = D

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

            if tau == 0:
                q = .5*lnD + .5*lnBs + lnAo
            elif tau == T-1:
                q = .5*lnBs + lnAo

            Qs[:,tau] = softmax(q)
            qs[i+1,:,tau] = Qs[:,tau]


    fig = figure()
    ax  = fig.add_subplot(1,1,1)
    ax.plot(qs[:,:,0], label=['$q,\\tau=0$','$q,\\tau=0$'])
    ax.plot(qs[:,:,1], label=['$q,\\tau=1$','$q,\\tau=1$'])
    ax.set_title('Example 1: Approximate posteriors (1 per edge per time point)')
    ax.set_xlabel('Message passing iterations')
    ax.legend()

    fig.savefig(join(args.figs,Path(__file__).stem))
    if args.show:
        show()

