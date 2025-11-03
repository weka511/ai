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
        self.context_names = ['Right-Attractive', 'Left-Attractive']
        self.choice_names = ['Start', 'Hint', 'Left Arm', 'Right Arm']
        self.choice_action_names = ['Start', 'Hint', 'Left Arm', 'Right Arm']
        self.state_names = ['Start', 'Hint-Left', 'Hint_Right', 'Left Arm', 'Right Arm']
        self.obs_names = ['Start', 'Hint-Left', 'Hint_Right', 'Left Arm', 'Right Arm']
        self.modalities = ['where', 'what']

    def create_A(self, epsilon=2.0 / 100.0):
        A = utils.obj_array(len(self.modalities))
        A[0] = np.empty((len(self.obs_names), len(self.context_names), len(self.choice_names)))
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
        A[1] = np.empty((3, len(self.context_names), len(self.choice_names)))
        A[1][:, 0, :] = np.array([[1.0, 1.0, 0.0, 0.0],
                                  [0.0, 0.0, epsilon, 1.0 - epsilon],
                                  [0.0, 0.0, 1.0 - epsilon, epsilon]])
        A[1][:, 1, :] = np.array([[1.0, 1.0, 0.0, 0.0],
                                  [0.0, 0.0, 1.0 - epsilon, epsilon],
                                  [0.0, 0.0, epsilon, 1.0 - epsilon]])
        return A

    def create_B(self):
        B = utils.obj_array(len(self.modalities))
        B[0] = np.zeros((len(self.choice_names), len(self.choice_names), len(self.choice_action_names)))
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
        B[1][:,:,0] = np.eye((len(self.context_names)))
        B[1][:,:,1] = np.eye((len(self.context_names)))
        B[1][:,:,2] = np.eye((len(self.context_names)))
        B[1][:,:,3] = np.eye((len(self.context_names)))
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

class Maze(Env):
    def __init__(self):
        pass

def parse_args():
    parser = ArgumentParser(__doc__)
    parser.add_argument('--show', default=False, action='store_true', help='Controls whether plot will be displayed')
    parser.add_argument('--figs', default='./figs', help='Location for storing plot files')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    factory = MDP_Factory()
    agent = Agent(A=factory.create_A(), B=factory.create_B(), C=factory.create_C(), D=factory.create_D())
    env = Env()

    with AxisIterator(figs=args.figs, title='Section 7.3: Decision Making and Planning as Inference',
                      show=args.show, name=Path(__file__).stem) as axes:
        pass

