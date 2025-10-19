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

'''Tutorial 1: Active inference from scratch'''


from argparse import ArgumentParser
from itertools import product
from os.path  import join
from pathlib  import Path
import numpy as np
from matplotlib import rc
from matplotlib.pyplot import figure, show
import seaborn as sns
from pymdp import utils

# Local application/library specific imports.

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

def plot_point_on_grid(state_vector, grid_locations):
    '''
    Plots the current location of the agent on the grid world
    '''
    state_index = np.where(state_vector)[0][0]
    y, x = grid_locations[state_index]
    grid_heatmap = np.zeros((3,3))
    grid_heatmap[y,x] = 1.0
    sns.heatmap(grid_heatmap, cbar = False, fmt='.0f')

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


if __name__=='__main__':
    rc('font',**{'family':'serif','serif':['Palatino']})
    rc('text', usetex=True)
    parser = ArgumentParser(__doc__)
    parser.add_argument('--seed',type=int,default=None,help='Seed for random number generator')
    parser.add_argument('--show', default=False, action='store_true', help='Controls whether plot will be displayed')
    parser.add_argument('--figs', default='./figs',                   help = 'Location for storing plot files')
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    fig = figure()

    my_categorical = rng.random(size=3)
    my_categorical = utils.norm_dist(my_categorical) # normalizes the distribution so it integrates to 1.0

    sampled_outcome = utils.sample(my_categorical)
    plot_beliefs(my_categorical, title_str = 'A random (unconditional) Categorical distribution',ax=fig.add_subplot(2,2,1))

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
    grid_locations = list(product(range(3), repeat = 2))
    plot_grid(grid_locations,ax=fig.add_subplot(2,2,2))
    n_states = len(grid_locations)
    n_observations = len(grid_locations)
    A = np.eye(n_observations, n_states)
    plot_likelihood(A, title_str = 'A matrix or $P(o|s)$',ax=fig.add_subplot(2,2,3))
    fig.tight_layout()
    fig.savefig(join(args.figs,Path(__file__).stem))

    if args.show:
        show()

