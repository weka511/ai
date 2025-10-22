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
    Tutorial 2: the Agent API
    https://pymdp-rtd.readthedocs.io/en/latest/notebooks/using_the_agent_class.html
'''

from argparse import ArgumentParser
# from itertools import product
import numpy as np
from pymdp import utils
# from pymdp.maths import softmax, spm_log_single as log_stable
# from pymdp.control import construct_policies
from tutorial_common import AxisIterator, plot_likelihood, plot_grid, plot_beliefs, plot_point_on_grid

def parse_args():
    parser = ArgumentParser(__doc__)
    parser.add_argument('--show', default=False, action='store_true', help='Controls whether plot will be displayed')
    parser.add_argument('--figs', default='./figs',                   help = 'Location for storing plot files')
    return parser.parse_args()

if __name__=='__main__':
    args = parse_args()

    with AxisIterator(n_rows=3,n_columns=3,figs=args.figs,title = 'Tutorial 2: the Agent API',show=args.show) as axes:

        context_names = ['Left-Better', 'Right-Better']
        choice_names = ['Start', 'Hint', 'Left Arm', 'Right Arm']

        num_states = [len(context_names), len(choice_names)]
        num_factors = len(num_states)

        context_action_names = ['Do-nothing']
        choice_action_names = ['Move-start', 'Get-hint', 'Play-left', 'Play-right']

        num_controls = [len(context_action_names), len(choice_action_names)]

        hint_obs_names = ['Null', 'Hint-left', 'Hint-right']
        reward_obs_names = ['Null', 'Loss', 'Reward']
        choice_obs_names = ['Start', 'Hint', 'Left Arm', 'Right Arm']

        num_obs = [len(hint_obs_names), len(reward_obs_names), len(choice_obs_names)]
        num_modalities = len(num_obs)

        A = utils.obj_array( num_modalities )

        p_hint = 0.7 # accuracy of the hint, according to the agent's generative model (how much does the agent trust the hint?)

        A_hint = np.zeros( (len(hint_obs_names), len(context_names), len(choice_names)) )

        for choice_id, choice_name in enumerate(choice_names):
            match(choice_name):
                case 'Start':
                    A_hint[0,:,choice_id] = 1.0
                case 'Hint':
                    A_hint[1:,:,choice_id] = np.array([[p_hint, 1.0 - p_hint],
                                                      [1.0 - p_hint,  p_hint]])
                case 'Left Arm':
                    A_hint[0,:,choice_id] = 1.0
                case 'Right Arm':
                    A_hint[0,:,choice_id] = 1.0

        A[0] = A_hint

        plot_likelihood(A[0][:,:,1], title_str = 'Probability of the two hint types, for the two game states',ax=next(axes))
