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

    def __init__(self, figsize=(14, 14), n_rows=3, n_columns=3, figs='figs', title='', show=False, name=Path(__file__).stem):
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

def infer_states(observation_index, A, prior):
    '''
    Implement inference here -- NOTE: prior is already passed in, so you don't need to do anything with the B matrix.
    This function has already been given P(s_t). The conditional expectation that creates 'today's prior',
    using 'yesterday's posterior', will happen *before calling* this function

    Parameters:
        observation_index
        A
        prior

    Returns:    qs
    '''

    log_likelihood = log_stable(A[observation_index, :])
    log_prior = log_stable(prior)
    return softmax(log_likelihood + log_prior)


if __name__ == '__main__':
    args = parse_args()
    A = 0.1 * np.array([[7, 1, 1, 1],
                        [1, 7, 1, 1],
                        [1, 1, 7, 1],
                        [1, 1, 1, 7]])

    B = 0.01 * np.array([[1, 1, 1, 97],
                         [97, 1, 1, 1],
                         [1, 97, 1, 1],
                         [1, 1, 97, 1]])

    D = np.array([1, 0, 0, 0])

    o = np.array([0, 1, 1, 3, 0])

    n_symbols,n_states = A.shape
    n_steps = 5
    P_states = np.zeros((n_states,n_steps))

    tau = 0
    P_states[:,tau] = D

    for tau in range(1,5):
        P_state_symbol = np.zeros((n_symbols,n_states))
        P_Trans = np.dot(B,P_states[:,tau-1].T)
        for i in range(n_symbols):
            for j in range(n_states):
                P_state_symbol[i,j] = P_Trans[j] * A[i,j]
        P_states[:,tau] = P_state_symbol[o[tau],:]/np.sum(P_state_symbol[o[tau],:])


    with AxisIterator(n_rows=2, n_columns=2, figs=args.figs, title='Figure 7.2',
                      show=args.show, name=Path(__file__).stem) as axes:

        ax = next(axes)

        ax = next(axes)
        sns.heatmap(P_states, annot=True, fmt=".1g", ax=ax,cmap='PuRd')
        ax.set_title('Retrospective Beliefs')
        ax.set_xlabel(r'$\tau$')
        ax.set_ylabel('State')

        ax = next(axes)

        ax = next(axes)
        ax.scatter(list(range(len(o))),o)
        ax.set_xticks(list(range(len(o))),[f'$o_{i}$' for i in range(1,len(o)+1)])
        ax.set_yticks([])
