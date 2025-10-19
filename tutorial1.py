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
from os.path  import join
from pathlib  import Path
import numpy as np
from matplotlib.pyplot import figure, show
import seaborn as sns
from pymdp import utils

# Local application/library specific imports.

def plot_likelihood(matrix, xlabels = list(range(9)), ylabels = list(range(9)), title_str = 'Likelihood distribution (A)'):
    '''
    Plots a 2-D likelihood matrix as a heatmap
    '''

    if not np.isclose(matrix.sum(axis=0), 1.0).all():
        raise ValueError('Distribution not column-normalized! Please normalize (ensure matrix.sum(axis=0) == 1.0 for all columns)')

    fig = figure(figsize = (6,6))
    ax = sns.heatmap(matrix, xticklabels = xlabels, yticklabels = ylabels, cmap = 'gray', cbar = False, vmin = 0.0, vmax = 1.0)
    ax.set_title(title_str)
    show()

def plot_grid(grid_locations, num_x = 3, num_y = 3 ):
    '''
    Plots the spatial coordinates of GridWorld as a heatmap, with each (X, Y) coordinate
    labeled with its linear index (its `state id`)
    '''
    grid_heatmap = np.zeros((num_x, num_y))
    for linear_idx, location in enumerate(grid_locations):
        y, x = location
        grid_heatmap[y, x] = linear_idx
    sns.set(font_scale=1.5)
    sns.heatmap(grid_heatmap, annot=True, cbar = False, fmt='.0f', cmap='crest')

def plot_point_on_grid(state_vector, grid_locations):
    '''
    Plots the current location of the agent on the grid world
    '''
    state_index = np.where(state_vector)[0][0]
    y, x = grid_locations[state_index]
    grid_heatmap = np.zeros((3,3))
    grid_heatmap[y,x] = 1.0
    sns.heatmap(grid_heatmap, cbar = False, fmt='.0f')

def plot_beliefs(belief_dist, title_str=''):
    '''
    Plot a categorical distribution or belief distribution, stored in the 1-D numpy vector `belief_dist`
    '''
    fig = figure()
    ax  = fig.add_subplot(1,1,1)
    if not np.isclose(belief_dist.sum(), 1.0):
        raise ValueError('Distribution not normalized! Please normalize')

    ax.grid(zorder=0)
    ax.bar(range(belief_dist.shape[0]), belief_dist, color='r', zorder=3)
    ax.set_xticks(range(belief_dist.shape[0]))
    ax.set_title(title_str)
    # plt.show()

if __name__=='__main__':
    parser = ArgumentParser(__doc__)
    parser.add_argument('--seed',type=int,default=None,help='Seed for random number generator')
    parser.add_argument('--show', default=False, action='store_true', help='Controls whether plot will be displayed')
    parser.add_argument('--figs', default='./figs',                   help = 'Location for storing plot files')
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    my_categorical = rng.random(size=3)
    my_categorical = utils.norm_dist(my_categorical) # normalizes the distribution so it integrates to 1.0

    print(my_categorical.reshape(-1,1)) # we reshape it to display it like a column vector
    print(f'Integral of the distribution: {round(my_categorical.sum(), 2)}')
    sampled_outcome = utils.sample(my_categorical)
    print(f'Sampled outcome: {sampled_outcome}')
    plot_beliefs(my_categorical, title_str = 'A random (unconditional) Categorical distribution')
    # fig = figure()
    # ax  = fig.add_subplot(1,1,1)

    # fig.savefig(join(args.figs,Path(__file__).stem))
    if args.show:
        show()

