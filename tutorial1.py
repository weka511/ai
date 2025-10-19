#!/usr/bin/env python

# Copyright (C) 2023 Simon Crase  simon@greenweaves.nz

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
    Tutorial 1: Active inference from scratch
    https://pymdp-rtd.readthedocs.io/en/latest/notebooks/active_inference_from_scratch.html
'''


from argparse import ArgumentParser
from itertools import product
from os.path  import join
from pathlib  import Path
from warnings import warn
import numpy as np
from matplotlib import rc
from matplotlib.pyplot import figure, show
import seaborn as sns
from pymdp import utils

class AxisIterator:
    def __init__(self,figsize=(12,12), n_rows = 3, n_columns = 3,figs='figs',title = ''):
        self.figsize=figsize
        self.n_rows = n_rows
        self.n_columns = n_columns
        self.seq = 0
        self.title = title
        self.figs = figs

    def __iter__(self):
        return self

    def __next__(self):
        if self.seq < self.n_rows*self.n_columns:
            self.seq += 1
        else:
            warn('Too many subplots')
        return self.fig.add_subplot(self.n_rows,self.n_columns,self.seq)

    def __enter__(self):
        self.fig = figure(figsize=self.figsize)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.fig.suptitle(self.title)
        self.fig.tight_layout()
        self.fig.savefig(join(self.figs,Path(__file__).stem))

def plot_likelihood(matrix, xlabels = list(range(9)), ylabels = list(range(9)), title_str = 'Likelihood distribution (A)',ax=None):
    '''
    Plots a 2-D likelihood matrix as a heatmap
    '''

    if not np.isclose(matrix.sum(axis=0), 1.0).all():
        raise ValueError('Distribution not column-normalized! Please normalize (ensure matrix.sum(axis=0) == 1.0 for all columns)')

    sns.heatmap(matrix, xticklabels = xlabels, yticklabels = ylabels, cmap = 'gray', cbar = False, vmin = 0.0, vmax = 1.0,ax=ax)
    ax.set_title(title_str)

def plot_grid(grid_locations, num_x = 3, num_y = 3,ax=None ):
    '''
    Plots the spatial coordinates of GridWorld as a heatmap, with each (X, Y) coordinate
    labeled with its linear index (its `state id`)
    '''
    grid_heatmap = np.zeros((num_x, num_y))
    for linear_idx, location in enumerate(grid_locations):
        y, x = location
        grid_heatmap[y, x] = linear_idx
    sns.set(font_scale=1.5)
    sns.heatmap(grid_heatmap, annot=True, cbar = False, fmt='.0f', cmap='crest',ax=ax)

def plot_point_on_grid(state_vector, grid_locations,ax=None):
    '''
    Plots the current location of the agent on the grid world
    '''
    state_index = np.where(state_vector)[0][0]
    y, x = grid_locations[state_index]
    grid_heatmap = np.zeros((3,3))
    grid_heatmap[y,x] = 1.0
    sns.heatmap(grid_heatmap, cbar = False, fmt='.0f',ax=ax)

def plot_beliefs(belief_dist, title_str='',ax=None):
    '''
    Plot a categorical distribution or belief distribution, stored in the 1-D numpy vector `belief_dist`
    '''
    if not np.isclose(belief_dist.sum(), 1.0):
        raise ValueError('Distribution not normalized! Please normalize')

    ax.grid(zorder=0)
    ax.bar(range(belief_dist.shape[0]), belief_dist, color='r', zorder=3)
    ax.set_xticks(range(belief_dist.shape[0]))
    ax.set_title(title_str)

def parse_args():
    parser = ArgumentParser(__doc__)
    parser.add_argument('--seed',type=int,default=None,help='Seed for random number generator')
    parser.add_argument('--show', default=False, action='store_true', help='Controls whether plot will be displayed')
    parser.add_argument('--figs', default='./figs',                   help = 'Location for storing plot files')
    return parser.parse_args()

def create_B_matrix( actions = ["UP", "DOWN", "LEFT", "RIGHT", "STAY"]):
    B = np.zeros( (len(grid_locations), len(grid_locations), len(actions)) )
    for action_id, action_label in enumerate(actions):
        for curr_state, grid_location in enumerate(grid_locations):
            y, x = grid_location
            match action_label:
                case "UP":
                    next_y = y - 1 if y > 0 else y
                    next_x = x
                case  "DOWN":
                    next_y = y + 1 if y < 2 else y
                    next_x = x
                case "LEFT":
                    next_x = x - 1 if x > 0 else x
                    next_y = y
                case "RIGHT":
                    next_x = x + 1 if x < 2 else x
                    next_y = y
                case  "STAY":
                    next_x = x
                    next_y = y
            new_location = (next_y, next_x)
            next_state = grid_locations.index(new_location)
            B[next_state, curr_state, action_id] = 1.0

    return B,actions

if __name__=='__main__':
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    rc('font',**{'family':'serif','serif':['Palatino']})
    rc('text', usetex=True)

    with AxisIterator(n_rows=4,n_columns=3,figs=args.figs,title = 'Tutorial 1: Active inference from scratch') as axes:

        #  create a simple categorical distribution
        my_categorical = rng.random(size=3)
        my_categorical = utils.norm_dist(my_categorical)

        sampled_outcome = utils.sample(my_categorical)
        plot_beliefs(my_categorical, title_str = f'Random categorical distribution, sample={sampled_outcome}', ax = next(axes))

        # Now let’s move onto conditional categorical distributions or likelihoods,
        # i.e. how we represent the distribution of one discrete random variable X,
        # conditioned on the settings of another discrete random variable Y.
        p_x_given_y = rng.random((3, 4))
        p_x_given_y = utils.norm_dist(p_x_given_y)
        print(p_x_given_y[:,0])
        print(p_x_given_y[:,0].reshape(-1,1))
        print(f'Integral of P(X|Y=0): {p_x_given_y[:,0].sum()}')
        p_y = np.array([0.75, 0.25]) # this is already normalized - you don't need to `utils.norm_dist()` it!

        # the columns here are already normalized - you don't need to `utils.norm_dist()` it!
        p_x_given_y = np.array([[0.6, 0.5],
                                [0.15, 0.41],
                                [0.25, 0.09]])

        print(p_y.round(3).reshape(-1,1))
        print(p_x_given_y.round(3))
        E_x_wrt_y = p_x_given_y.dot(p_y)
        print(E_x_wrt_y)
        print(f'Integral: {E_x_wrt_y.sum().round(3)}')

        # A simple environment: Grid-world
        # Create  the grid locations in the form of a list of (Y, X) tuples
        grid_locations = list(product(range(3), repeat = 2))
        plot_grid(grid_locations,ax = next(axes))

        n_states = len(grid_locations)
        n_observations = len(grid_locations)

        # The A matrix or $A=P(o|s)
        # Create an umambiguous or 'noise-less' mapping between hidden states and observations
        A = np.eye(n_observations, n_states)
        plot_likelihood(A, title_str = '$A=P(o|s)$',ax = next(axes))
        A_noisy = A.copy()
        A_noisy[0,0] = 1.0/3.0
        A_noisy[1,0] = 1.0/3.0
        A_noisy[3,0] = 1.0/3.0
        plot_likelihood(A_noisy, title_str = 'Blurred A matrix',ax = next(axes))

        my_A_noisy = A_noisy.copy()

        # locations 3 and 7 are the nearest neighbours to location 6
        my_A_noisy[3,6] = 1.0 / 3.0
        my_A_noisy[6,6] = 1.0 / 3.0
        my_A_noisy[7,6] = 1.0 / 3.0
        plot_likelihood(my_A_noisy, title_str = 'Noisy A matrix',
                        ax = next(axes))

        B,actions = create_B_matrix()

        starting_location = (1,0)
        state_index = grid_locations.index(starting_location)
        starting_state = utils.onehot(state_index, n_states)
        plot_point_on_grid(starting_state, grid_locations, ax = next(axes))

        plot_beliefs(starting_state, "Categorical distribution over the starting state", ax = next(axes))

        right_action_idx = actions.index("RIGHT")
        next_state = B[:,:, right_action_idx].dot(starting_state) # input the indices to the B matrix
        plot_point_on_grid(next_state, grid_locations, ax = next(axes))

        prev_state = next_state.copy()
        down_action_index = actions.index("DOWN")
        next_state = B[:,:,down_action_index].dot(prev_state)
        plot_point_on_grid(next_state, grid_locations,ax = next(axes))

        #  preferences over observations
        C = np.zeros(n_observations)
        desired_location = (2,2) # choose a desired location
        desired_location_index = grid_locations.index(desired_location) # get the linear index of the grid location, in terms of 0 through 8

        C[desired_location_index] = 1.0
        plot_beliefs(C, title_str = "Preferences over observations", ax = next(axes))

        #  prior belief over hidden states at the first timestep
        D = utils.onehot(0, n_states)

        # demonstrate hwo belief about initial state can also be uncertain / spread among different possible initial states
        # alternative, where you have a degenerate/noisy prior belief
        # D = utils.norm_dist(np.ones(n_states))

        """ Let's look at the prior over hidden states """
        plot_beliefs(D, title_str = "Prior beliefs over states", ax = next(axes))

    if args.show:
        show()

