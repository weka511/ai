#!/usr/bin/env python
# -*- coding: utf-8 -*-

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
    Common code for tutorials
'''

import numpy as np
from matplotlib import rc
from matplotlib.pyplot import figure, show
import seaborn as sns


def plot_likelihood(matrix, xlabels=None, ylabels=None, title_str='Likelihood distribution (A)', ax=None,cbar=False):
	'''
	Plots a 2-D likelihood matrix as a heatmap
	'''
	m, n = matrix.shape
	if xlabels == None:
		xlabels = list(range(m))
	if ylabels == None:
		ylabels = list(range(n))
	if not np.isclose(matrix.sum(axis=0), 1.0).all():
		raise ValueError('Distribution not column-normalized! Please normalize (ensure matrix.sum(axis=0) == 1.0 for all columns)')

	sns.heatmap(matrix, xticklabels=xlabels, yticklabels=ylabels, cmap='viridis', cbar=cbar, vmin=0.0, vmax=1.0, ax=ax)
	ax.set_title(title_str, fontsize=8)
	ax.tick_params(axis='x', labelsize=8)
	ax.tick_params(axis='y', labelsize=8)


def plot_grid(grid_locations, num_x=3, num_y=3, ax=None):
	'''
	Plots the spatial coordinates of GridWorld as a heatmap, with each (X, Y) coordinate
	labeled with its linear index (its `state id`)
	'''
	grid_heatmap = np.zeros((num_x, num_y))
	for linear_idx, location in enumerate(grid_locations):
		y, x = location
		grid_heatmap[y, x] = linear_idx
	sns.set(font_scale=1.5)
	sns.heatmap(grid_heatmap, annot=True, cbar=False, fmt='.0f', cmap='viridis', ax=ax)
	ax.tick_params(axis='x', labelsize=8)
	ax.tick_params(axis='y', labelsize=8)


def plot_point_on_grid(state_vector, grid_locations, ax=None, title='Current location'):
	'''
	Plots the current location of the agent on the grid world
	'''
	state_index = np.nonzero(state_vector)[0][0]
	y, x = grid_locations[state_index]
	grid_heatmap = np.zeros((3, 3))
	grid_heatmap[y, x] = 1.0
	sns.heatmap(grid_heatmap, cbar=False, fmt='.0f', ax=ax, cmap='viridis')
	ax.set_title(title, fontsize=8)
	ax.tick_params(axis='x', labelsize=8)
	ax.tick_params(axis='y', labelsize=8)


def plot_beliefs(belief_dist, title_str='', ax=None):
	'''
	Plot a categorical distribution or belief distribution, stored in the 1-D numpy vector `belief_dist`
	'''
	if not np.isclose(belief_dist.sum(), 1.0):
		raise ValueError('Distribution not normalized! Please normalize')

	ax.grid(zorder=0)
	ax.bar(range(belief_dist.shape[0]), belief_dist, color='r', zorder=3)
	ax.set_xticks(range(belief_dist.shape[0]))
	ax.set_title(title_str, fontsize=8)
	ax.tick_params(axis='x', labelsize=8)
	ax.tick_params(axis='y', labelsize=8)
