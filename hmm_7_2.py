#!/usr/bin/env python

# Copyright (C) 2025 Simon Crase  simon@greenweaves.nz

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

'''HMM Example from Section 7-2'''


# Standard library imports.

from argparse import ArgumentParser
from os.path import join
from pathlib import Path
from matplotlib.pyplot import figure, show
from matplotlib import rc
import numpy as np
from pymdp.maths import softmax, spm_log_single as log_stable
import seaborn as sns

def parse_args():
    parser = ArgumentParser(__doc__)
    parser.add_argument('--show', default=False, action='store_true', help='Controls whether plot will be displayed')
    parser.add_argument('--figs', default='./figs', help='Location for storing plot files')
    return parser.parse_args()

class AxisIterator:
    '''
    This class creates subplots as needed
    '''
    def __init__(self, figsize=(8, 8), n_rows=2, n_columns=2, figs='figs', title='', show=False, name=Path(__file__).stem):
        self.figsize = figsize
        self.n_rows = n_rows
        self.n_columns = n_columns
        self.seq = 0
        self.title = title
        self.figs = figs
        self.show = show
        self.name = name

    def __iter__(self):
        return self

    def __next__(self):
        '''
        Used to supply subplots
        '''
        if self.seq < self.n_rows * self.n_columns:
            self.seq += 1
        else:
            warn('Too many subplots')

        return self.fig.add_subplot(self.n_rows, self.n_columns, self.seq)

    def __enter__(self):
        rc('font', **{'family': 'serif',
                              'serif': ['Palatino'],
                    'size': 8})
        rc('text', usetex=True)
        self.fig = figure(figsize=self.figsize)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.fig.suptitle(self.title, fontsize=10)
        self.fig.tight_layout(pad=3,h_pad=4,w_pad=3)
        self.fig.savefig(join(self.figs, self.name))
        if self.show:
            show()

def create_P_state_symbol(prior,A,B):
    '''
    Create a matrix of containing the probability of being in each state and emitting each symbol,
    given the probabilties of being in each state from the prvious step.

    Parameters:
        prior   Vector of probabilties of each state from previous time step
        A       Emission matrix, each row being the probabilty of one particular output from each state
        B       Transition matrix, each row being the probabilty of one particular state
                as a transition from previous state
    '''
    n_symbols,n_states = A.shape
    product = np.zeros((n_symbols,n_states))
    P_Trans = np.dot(B,prior) # Probabilities of each state, given prior
    for i in range(n_symbols):
        for j in range(n_states):
            product[i,j] = P_Trans[j] * A[i,j]
    return product

def create_beliefs(A,B,D,o):
    '''
    Calculate the Observer's beliefs, i.e. her estimate of the probability
    of being in each hidden state at a particular time, given the observations.

    Parameters:
        A
        B
        D
        o

    Returns:
        Matrix of probabilties: the rows represent states, and the columns times
    '''
    n_symbols,n_states = A.shape
    n_steps = len(o)

    Beliefs = np.zeros((n_states,n_steps))

    tau = 0
    Beliefs[:,tau] = D

    # Evolve state, based on observed symbols

    for tau in range(1,n_steps):
        P_state_symbol = create_P_state_symbol(Beliefs[:,tau-1].T,A,B)
        Beliefs[:,tau] = P_state_symbol[o[tau],:]/np.sum(P_state_symbol[o[tau],:])

    return Beliefs

if __name__ == '__main__':
    args = parse_args()

    # Emission matrix, each row being the probabilty of one particular output from each state
    A = 0.1 * np.array([[7, 1, 1, 1],
                        [1, 7, 1, 1],
                        [1, 1, 7, 1],
                        [1, 1, 1, 7]])

    # Transition matrix, each row being the probabilty of one particular state as a transition from previous state

    B = 0.01 * np.array([[1, 1, 1, 97],
                         [97, 1, 1, 1],
                         [1, 97, 1, 1],
                         [1, 1, 97, 1]])

    # Prior probabilities for each state

    D = np.array([1, 0, 0, 0])

    # Observed symbols

    o = np.array([0, 1, 1, 3, 0])



    Beliefs = create_beliefs(A,B,D,o)

    with AxisIterator(figs=args.figs, title='Figure 7.2: Perceptual Processing',
                      show=args.show, name=Path(__file__).stem) as axes:

        ax = next(axes)

        ax = next(axes)
        sns.heatmap(Beliefs, annot=True, fmt=".1g", ax=ax,cmap='PuRd')
        ax.set_title('Retrospective Beliefs based on observed symbols')
        ax.set_xlabel(r'$\tau$')
        ax.set_ylabel('State')

        ax = next(axes)

        ax = next(axes)
        ax.scatter(list(range(len(o))),o)
        ax.set_xticks(list(range(len(o))),[f'$o_{i}$' for i in range(1,len(o)+1)])
        ax.set_title('Observed symbols')
        ax.set_yticks([0,1,2,3])
