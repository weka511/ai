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
from os.path import join
from pathlib import Path
import numpy as np
from pymdp import utils
from pymdp.maths import softmax
from pymdp.agent import Agent
from tutorial_common import AxisIterator, plot_likelihood, plot_grid, plot_beliefs, plot_point_on_grid

class TwoArmedBandit(object):

    def __init__(self, context = None, p_hint = 1.0, p_reward = 0.8):

        self.context_names = ['Left-Better', 'Right-Better']

        if context == None:
            self.context = self.context_names[utils.sample(np.array([0.5, 0.5]))] # randomly sample which bandit arm is better (Left or Right)
        else:
            self.context = context

        self.p_hint = p_hint
        self.p_reward = p_reward

        self.reward_obs_names = ['Null', 'Loss', 'Reward']
        self.hint_obs_names = ['Null', 'Hint-left', 'Hint-right']

    def step(self, action):
        match(action):
            case 'Move-start':
                observed_hint = 'Null'
                observed_reward = 'Null'
                observed_choice = 'Start'
            case 'Get-hint':
                if self.context == 'Left-Better':
                    observed_hint = self.hint_obs_names[utils.sample(np.array([0.0, self.p_hint, 1.0 - self.p_hint]))]
                if self.context == 'Right-Better':
                    observed_hint = self.hint_obs_names[utils.sample(np.array([0.0, 1.0 - self.p_hint, self.p_hint]))]
                observed_reward = 'Null'
                observed_choice = 'Hint'
            case 'Play-left':
                observed_hint = 'Null'
                observed_choice = 'Left Arm'
                if self.context == 'Left-Better':
                    observed_reward = self.reward_obs_names[utils.sample(np.array([0.0, 1.0 - self.p_reward, self.p_reward]))]
                if self.context == 'Right-Better':
                    observed_reward = self.reward_obs_names[utils.sample(np.array([0.0, self.p_reward, 1.0 - self.p_reward]))]
            case 'Play-right':
                observed_hint = 'Null'
                observed_choice = 'Right Arm'
                if self.context == 'Right-Better':
                    observed_reward = self.reward_obs_names[utils.sample(np.array([0.0, 1.0 - self.p_reward, self.p_reward]))]
                if self.context == 'Left-Better':
                    observed_reward = self.reward_obs_names[utils.sample(np.array([0.0, self.p_reward, 1.0 - self.p_reward]))]

        return [observed_hint, observed_reward, observed_choice]

def parse_args():
    parser = ArgumentParser(__doc__)
    parser.add_argument('--show', default=False, action='store_true', help='Controls whether plot will be displayed')
    parser.add_argument('--figs', default='./figs', help = 'Location for storing plot files')
    return parser.parse_args()

if __name__=='__main__':
    args = parse_args()

    with AxisIterator(n_rows=3,n_columns=3,figs=args.figs,title = 'Tutorial 2: the Agent API',
                      show=args.show,name=Path(__file__).stem) as axes:

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

        p_reward = 0.8 # probability of getting a rewarding outcome, if you are sampling the more rewarding bandit

        A_reward = np.zeros((len(reward_obs_names), len(context_names), len(choice_names)))

        for choice_id, choice_name in enumerate(choice_names):
            match (choice_name):
                case 'Start':
                    A_reward[0,:,choice_id] = 1.0
                case'Hint':
                    A_reward[0,:,choice_id] = 1.0
                case 'Left Arm':
                    A_reward[1:,:,choice_id] = np.array([ [1.0-p_reward, p_reward],
                                                      [p_reward, 1.0-p_reward]])
                case 'Right Arm':
                    A_reward[1:, :, choice_id] = np.array([[ p_reward, 1.0- p_reward],
                                                   [1- p_reward, p_reward]])

        A[1] = A_reward

        plot_likelihood(A[0][:,:,1], title_str = 'Probability of the two hint types, for the two game states',ax=next(axes))

        A_choice = np.zeros((len(choice_obs_names), len(context_names), len(choice_names)))

        for choice_id in range(len(choice_names)):
            A_choice[choice_id, :, choice_id] = 1.0

        A[2] = A_choice

        plot_likelihood(A[2][:,0,:], title_str='Mapping between sensed states and true states',ax=next(axes))

        B = utils.obj_array(num_factors)

        B_context = np.zeros( (len(context_names), len(context_names), len(context_action_names)) )

        B_context[:,:,0] = np.eye(len(context_names))

        B[0] = B_context

        B_choice = np.zeros( (len(choice_names), len(choice_names), len(choice_action_names)) )

        for choice_i in range(len(choice_names)):

            B_choice[choice_i, :, choice_i] = 1.0

        B[1] = B_choice

        C = utils.obj_array_zeros(num_obs)

        C_reward = np.zeros(len(reward_obs_names))
        C_reward[1] = -4.0
        C_reward[2] = 2.0

        C[1] = C_reward

        plot_beliefs(softmax(C_reward), title_str = 'Prior preferences',ax=next(axes))

        D = utils.obj_array(num_factors)

        D_context = np.array([0.5,0.5])

        D[0] = D_context

        D_choice = np.zeros(len(choice_names))

        D_choice[choice_names.index('Start')] = 1.0

        D[1] = D_choice

        print(f'Beliefs about which arm is better: {D[0]}')
        print(f'Beliefs about starting location: {D[1]}')

        my_agent = Agent(A = A, B = B, C = C, D = D)
