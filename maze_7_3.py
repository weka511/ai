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
    Example from Section 7.3 Decision Making and Planning as Inference

    Code adapted from Tutorial 2: the Agent API
    https://pymdp-rtd.readthedocs.io/en/latest/notebooks/using_the_agent_class.html
'''

from argparse import ArgumentParser
from pathlib import Path
import numpy as np
from pymdp import utils
from pymdp.agent import Agent
from pymdp.envs import Env
from pymdp.maths import softmax, spm_log_single as log_stable, spm_norm as norm
from ai import AxisIterator


class MDP_Factory:
    '''
    This class creates the A, B, C, and D matrices for the maze example
    '''
    def __init__(self):
        self.context_names = ['Right Attractive', 'Left Attractive']
        self.location_names = ['Start', 'Bottom', 'Left', 'Right']
        self.choice_action_names = ['Move Start', 'Move Bottom', 'Move Left', 'Move Right']
        self.location_obs_names = ['At Start', 'At Bottom: Left Attractive', 'At Bottom: Right Attractive', 'At Left', 'At Right']
        self.modalities = ['where', 'what']

    def create_A(self, probability_hint_wrong=2.0 / 100.0):
        A = utils.obj_array(len(self.modalities))
        A[0] = np.empty((len(self.location_obs_names), len(self.context_names), len(self.location_names)))
        A[0][:, 0, :] = np.array([[1.0, 0.0, 0.0, 0.0],
                                  [0.0, 0.0, 0.0, 0.0],
                                  [0.0, 1.0, 0.0, 0.0],
                                  [0.0, 0.0, 1.0, 0.0],
                                  [0.0, 0.0, 0.0, 1.0]])
        A[0][:, 1, :] = np.array([[1.0, 0.0, 0.0, 0.0],
                                  [0.0, 1.0, 0.0, 0.0],
                                  [0.0, 0.0, 0.0, 0.0],
                                  [0.0, 0.0, 1.0, 0.0],
                                  [0.0, 0.0, 0.0, 1.0]])
        A[1] = np.empty((3, len(self.context_names), len(self.location_names)))
        A[1][:, 0, :] = np.array([[1.0, 1.0, 0.0, 0.0],
                                  [0.0, 0.0, probability_hint_wrong, 1.0 - probability_hint_wrong],
                                  [0.0, 0.0, 1.0 - probability_hint_wrong, probability_hint_wrong]])
        A[1][:, 1, :] = np.array([[1.0, 1.0, 0.0, 0.0],
                                  [0.0, 0.0, 1.0 - probability_hint_wrong, probability_hint_wrong],
                                  [0.0, 0.0, probability_hint_wrong, 1.0 - probability_hint_wrong]])
        return A

    def create_B(self):
        B = utils.obj_array(len(self.modalities))
        B[0] = np.zeros((len(self.location_names), len(self.location_names), len(self.choice_action_names)))
        B[0][:, :, 0] = np.array([[1.0, 1.0, 0.0, 0.0],
                                  [0.0, 0.0, 0.0, 0.0],
                                  [0.0, 0.0, 1.0, 0.0],
                                  [0.0, 0.0, 0.0, 1.0]])
        B[0][:, :, 1] = np.array([[0.0, 0.0, 0.0, 0.0],
                                  [1.0, 1.0, 0.0, 0.0],
                                  [0.0, 0.0, 1.0, 0.0],
                                  [0.0, 0.0, 0.0, 1.0]])
        B[0][:, :, 2] = np.array([[0.0, 0.0, 0.0, 0.0],
                                  [0.0, 0.0, 0.0, 0.0],
                                  [1.0, 1.0, 1.0, 0.0],
                                  [0.0, 0.0, 0.0, 1.0]])
        B[0][:, :, 3] = np.array([[0.0, 0.0, 0.0, 0.0],
                                  [0.0, 0.0, 0.0, 0.0],
                                  [0.0, 0.0, 1.0, 0.0],
                                  [1.0, 1.0, 0.0, 1.0]])
        B[1] = np.zeros((len(self.context_names), len(self.context_names), len(self.choice_action_names)))
        B[1][:, :, 0] = np.eye((len(self.context_names)))
        B[1][:, :, 1] = np.eye((len(self.context_names)))
        B[1][:, :, 2] = np.eye((len(self.context_names)))
        B[1][:, :, 3] = np.eye((len(self.context_names)))
        return B

    def create_C(self):
        C = utils.obj_array(len(self.modalities))
        C[0] = softmax(np.c_[[-1.0, 0.0, 0.0, 0.0, 0.0]])
        C[1] = softmax(np.c_[[0.0, 6.0, -6.0]])
        return C

    def create_D(self):
        D = utils.obj_array(len(self.modalities))
        D[0] = np.c_[[1.0, 0.0, 0.0, 0.0]]
        D[1] = norm(np.c_[[1.0, 1.0]])
        return D


class MazeEnvironment(Env):
    '''
    The class represents the generative process for the maze
    '''

    def __init__(self, factory, rng=np.random.default_rng(),probability_hint_wrong=2.0 / 100.0):
        '''
        Assign context (left or right) at random, and place mouse at starting position.

        Create a table of allowable transitions by or-ing the matrices from Figure 7.6,
        which  makes Left and Right absorbing states.
        '''
        self.rng = rng
        self.location = 0
        self.context = self.rng.choice(len(factory.context_names))
        self.AllowableTransitions = np.array([[1, 1, 0, 0],
                                              [1, 1, 0, 0],
                                              [1, 1, 1, 0],
                                              [1, 1, 0, 1]],dtype=bool)
        self.probability_hint_wrong = probability_hint_wrong

    def step(self, action):
        '''
        Update to position in response to an action
        '''
        if self.AllowableTransitions[self.location,action]:
            self.location = action

        match self.location: #  ['Start', 'Bottom', 'Left', 'Right']
            case 0:      # Start
                return 0,0 # At Start
            case 1:      # Bottom
                if self.context == 0: # Right Attractive
                    if self.rng.uniform() < self.probability_hint_wrong:
                        return 1,0 # At Bottom: Left Attractive
                    else:
                        return 2,0 #  At Bottom: Right Attractive
                else:                 #  Left Attractive
                    if self.rng.uniform() < self.probability_hint_wrong:
                        return 2,0 #  At Bottom: Right Attractive
                    else:
                        return 1,1 # At Bottom: Left Attractive
            case 2: # Left
                if self.context == 0: # Right Attractive
                    return 3,2   # At Left
                else:
                    return 3,1
            case 3: # Right
                if self.context == 0: # Right Attractive
                    return 4,1   # At Left
                else:
                    return 4,2


def parse_args():
    parser = ArgumentParser(__doc__)
    parser.add_argument('--show', default=False, action='store_true', help='Controls whether plot will be displayed')
    parser.add_argument('--figs', default='./figs', help='Location for storing plot files')
    parser.add_argument('--seed', default=None, type=int, help='Initialize random number generator')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    factory = MDP_Factory()
    agent = Agent(A=factory.create_A(), B=factory.create_B(), C=factory.create_C(), D=factory.create_D())
    env = MazeEnvironment(factory, rng=rng)
    T = 2
    action = 0
    for t in range(T):
        o = env.step(action)
        qs = agent.infer_states(o)


    # with AxisIterator(figs=args.figs, title='Section 7.3: Decision Making and Planning as Inference',
                      # show=args.show, name=Path(__file__).stem) as axes:
        # pass

