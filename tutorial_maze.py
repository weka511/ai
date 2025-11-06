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

'''Active Inference Demo: T-Maze Environment'''


from argparse import ArgumentParser
from pathlib import Path
import numpy as np
from pymdp.maths import softmax, spm_log_single as log_stable
from pymdp.agent import Agent
from pymdp.utils import plot_beliefs, plot_likelihood
from pymdp.envs import TMazeEnv
from ai import AxisIterator
from tutorial_common import plot_likelihood, plot_grid, plot_beliefs, plot_point_on_grid

def parse_args():
    parser = ArgumentParser(__doc__)
    parser.add_argument('--show', default=False, action='store_true', help='Controls whether plot will be displayed')
    parser.add_argument('--figs', default='./figs', help='Location for storing plot files')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    reward_probabilities = [0.98, 0.02]
    env = TMazeEnv(reward_probs = reward_probabilities)
    A_gp = env.get_likelihood_dist()
    with AxisIterator(figs=args.figs, title=__doc__,
                      show=args.show, name=Path(__file__).stem) as axes:
        plot_likelihood(A_gp[1][:,:,0],'Reward Right',ax=next(axes),cbar=True)


