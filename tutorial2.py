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

'''Tutorial 2: the Agent API'''

from argparse import ArgumentParser
from os.path  import join
from pathlib  import Path
from matplotlib.pyplot import figure, show
import numpy as np
import seaborn as sns
from pymdp import utils
from pymdp.maths import softmax
from pymdp.agent import Agent

def plot_likelihood(matrix, title_str = "Likelihood distribution (A)"):
    '''
    Plots a 2-D likelihood matrix as a heatmap
    '''

    if not np.isclose(matrix.sum(axis=0), 1.0).all():
        raise ValueError("Distribution not column-normalized! Please normalize (ensure matrix.sum(axis=0) == 1.0 for all columns)")

    fig = plt.figure(figsize = (6,6))
    ax = sns.heatmap(matrix, cmap = 'gray', cbar = False, vmin = 0.0, vmax = 1.0)
    plt.title(title_str)
    plt.show()


def plot_beliefs(belief_dist, title_str=""):
    '''
    Plot a categorical distribution or belief distribution, stored in the 1-D numpy vector `belief_dist`
    '''

    if not np.isclose(belief_dist.sum(), 1.0):
        raise ValueError("Distribution not normalized! Please normalize")

    plt.grid(zorder=0)
    plt.bar(range(belief_dist.shape[0]), belief_dist, color='r', zorder=3)
    plt.xticks(range(belief_dist.shape[0]))
    plt.title(title_str)
    plt.show()

if __name__=='__main__':
    parser = ArgumentParser(__doc__)
    parser.add_argument('--show', default=False, action='store_true', help='Controls whether plot will be displayed')
    parser.add_argument('--figs', default='./figs',                   help = 'Location for storing plot files')
    args = parser.parse_args()

    fig = figure()
    ax  = fig.add_subplot(1,1,1)

    fig.savefig(join(args.figs,Path(__file__).stem))
    if args.show:
        show()

