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

'''
    Hidden Markov Model Example for musical notes

    Produce simulated inference plots to illustrate belief updating
    based on the generative model outlined in Section 7-2 Perceptual Processing
'''

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
        self.fig.suptitle(self.title, fontsize=12)
        self.fig.tight_layout(pad=3, h_pad=4, w_pad=3)
        self.fig.savefig(join(self.figs, self.name))
        if self.show:
            show()


def create_P_state_symbol(prior, A, B):
    '''
    Create a matrix of containing the probability of being in each state and emitting each symbol,
    given the probabilties of being in each state from the prvious step.

    Parameters:
        prior   Vector of probabilties of each state from previous time step
        A       Emission matrix, each row being the probabilty of one particular output from each state
        B       Transition matrix, each row being the probabilty of one particular state
                as a transition from previous state
    '''
    n_symbols, n_states = A.shape
    P_state_symbol = np.zeros((n_symbols, n_states))
    P_Trans = np.dot(B, prior) # Probabilities of each state, given prior
    for i in range(n_symbols):
        for j in range(n_states):
            P_state_symbol[i, j] = P_Trans[j] * A[i, j]
    return P_state_symbol


def create_beliefs(A, B, D, o):
    '''
    Calculate the Observer's beliefs, i.e. her estimate of the probability
    of being in each hidden state at a particular time, given the observations.

    Parameters:
        A     Emission matrix, the probability of one particular output given the state.
        B     Transition matrix, each row being the probabilty of one particular state as a transition from previous state
        D     Prior probabilities for each state
        o     The symbols actually observed at each time step

    Returns:
        Matrix of probabilties: the rows represent states, and the columns times
    '''
    n_symbols, n_states = A.shape
    n_steps = len(o)

    Beliefs = np.zeros((n_states, n_steps))

    tau = 0
    Beliefs[:, tau] = D

    # Evolve state, based on observed symbols

    for tau in range(1, n_steps):
        P_state_symbol = create_P_state_symbol(Beliefs[:, tau - 1].T, A, B)
        Beliefs[:, tau] = P_state_symbol[o[tau], :] / np.sum(P_state_symbol[o[tau], :])

    return Beliefs


def infer(A, B, D, o):
    '''
    Infer states using equation 4.13

    Parameters:
        A     Emission matrix, the probability of one particular output given the state.
        B     Transition matrix, each row being the probabilty of one particular state as a transition from previous state
        D     Prior probabilities for each state
        o     The symbols actually observed at each time step
    '''
    n_symbols, n_states = A.shape
    n_steps = len(o)
    ln_A = log_stable(A)
    ln_B = log_stable(B)
    s = np.zeros((n_states, n_steps))
    s[:, 0] = D
    v = np.zeros((n_states, n_steps))
    v[:, 0] = log_stable(s[:, 0])
    dv = np.zeros((n_states, n_steps))
    o_tau = np.zeros((n_symbols, n_steps))
    for tau in range(n_steps):
        o_tau[o[tau], tau] = 1

    tau = 0

    for tau in range(1, n_steps):
        dv[:,tau] = np.dot(ln_A, o_tau[:, tau - 1]) + 2 * np.dot(ln_B, s[:, tau - 1].T) - log_stable(s[:, tau - 1])
        v[:, tau] = v[:, tau - 1] + dv[:,tau]
        s[:, tau] = softmax(v[:, tau])

    return s, dv


if __name__ == '__main__':
    args = parse_args()

    # Emission matrix, the probability of one particular output given the state.
    # Each row represents an output, and the coumns represent states
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

    # The symbols actually observed at each time step

    o = np.array([0, 1, 1, 3, 0])

    s, dv = infer(A, B, D, o)
    n_states, n_steps = s.shape

    with AxisIterator(figs=args.figs, title='Figure 7.2: Perceptual Processing',
                      show=args.show, name=Path(__file__).stem) as axes:

        ax = next(axes)
        for i in range(n_states):
            ax.plot(s[i, :], label=f'{i}')
        ax.legend(title='State')
        ax.set_title('Beliefs')
        ax.set_xticks(np.arange(0, n_steps))
        ax.set_xlabel(r'$\tau$')
        ax.set_ylabel(r'$s_\tau$')

        ax = next(axes)
        sns.heatmap(create_beliefs(A, B, D, o), annot=True, fmt=".1g", ax=ax, cmap='PuRd')
        ax.set_title('Retrospective Beliefs based on observed symbols')
        ax.set_xlabel(r'$\tau$')

        ax = next(axes)
        for i in range(4):
            ax.plot(dv[i, :], label=f'{i}')
        ax.legend(title='State')
        ax.set_title('Free Energy Gradients')
        ax.set_xlabel(r'$\tau$')
        ax.set_ylabel(r'$\epsilon_\tau$')
        ax.set_xticks(np.arange(0, n_steps))

        ax = next(axes)
        ax.scatter(list(range(len(o))), o)
        ax.set_xlabel(r'$\tau$')
        ax.set_xticks(np.arange(0, len(o)))
        ax.set_title('Observed symbols')
        ax.set_yticks([0, 1, 2, 3])
        ax.set_ylabel('Symbol')

