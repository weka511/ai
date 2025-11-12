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
from enum import IntEnum
from pathlib import Path
import numpy as np
from pymdp import utils
from pymdp.agent import Agent
from pymdp.envs import Env
from pymdp.maths import softmax, spm_norm as norm, spm_log_single as log_stable


class Context(IntEnum):
    '''
    Determines where attractive stimulus is located
    '''
    RIGHT_ATTRACTIVE = 0
    LEFT_ATTRACTIVE = 1


class Location(IntEnum):
    '''
    Location of Mouse
    '''
    START = 0
    BOTTOM = 1
    LEFT = 2
    RIGHT = 3


class Move(IntEnum):
    '''
    Controls where Mouse will move.
    It is numerically equal to new location
    '''
    START = 0
    BOTTOM = 1
    LEFT = 2
    RIGHT = 3


class LocationObservation(IntEnum):
    '''
    This is what mouse observes
    '''
    AT_START = 0
    AT_BOTTOM_LEFT_ATTRACTIVE = 1
    AT_BOTTOM_RIGHT_ATTRACTIVE = 2
    AT_LEFT = 3
    AT_RIGHT = 4


class Modality(IntEnum):
    '''
    There are two modalities, i.e. two types of observations, location and stimulus
    '''
    WHERE = 0
    WHAT = 1


class Stimulus(IntEnum):
    '''
    Stimulus at a location
    '''
    NONE = 0
    ATTRACTIVE = 1
    AVERSIVE = 2


class Hint(IntEnum):
    '''
    Suggested location for attractive stimulus, which may be unreliable
    '''
    HINT_NONE = 0
    HINT_LEFT = 1
    HINT_RIGHT = 2


class MazeFactory:
    '''
    This class creates the A, B, C, and D matrices for the maze example
    '''

    def create_A(self, a = 0.98, b = 0.02):
        '''
        Set up Likelihood Matrix
        '''
        A = utils.obj_array(len(Modality))
        A[Modality.WHERE] = np.empty((len(LocationObservation), len(Location), len(Context)))
        A[Modality.WHERE][:, :, Context.RIGHT_ATTRACTIVE] = np.array([[1, 0, 0, 0],
                                                                      [0, 0, 0, 0],
                                                                      [0, 1, 0, 0],
                                                                      [0, 0, 1, 0],
                                                                      [0, 0, 0, 1]],
                                                                     dtype=float)
        A[Modality.WHERE][:, :, Context.LEFT_ATTRACTIVE] = np.array([[1, 0, 0, 0],
                                                                     [0, 1, 0, 0],
                                                                     [0, 0, 0, 0],
                                                                     [0, 0, 1, 0],
                                                                     [0, 0, 0, 1]],
                                                                    dtype=float)

        A[Modality.WHAT] = np.empty((len(Hint), len(Location), len(Context)))
        A[Modality.WHAT][:, :, Context.RIGHT_ATTRACTIVE] = np.array([[1, 1, 0, 0],
                                                                     [0, 0, a, b],
                                                                     [0, 0, b, a]],
                                                                    dtype=float)
        A[Modality.WHAT][:, :, Context.LEFT_ATTRACTIVE] = np.array([[1, 1, 0, 0],
                                                                    [0, 0, b, a],
                                                                    [0, 0, a, b]],
                                                                    dtype=float)

        return A

    def create_B(self,n_policies=4):
        '''
        Set up Transition Probabilities
        '''
        B = utils.obj_array(len(Modality))
        B[Modality.WHERE] = self.create_B_location(n_policies=n_policies)
        B[Modality.WHAT] = np.zeros((len(Context), len(Context), n_policies))
        for i in range(n_policies):
            B[Modality.WHAT][:, :, i] = np.eye((len(Context)))
        return B

    def create_C(self,c=6.0):
        '''
        Set up Prior Preferences
        Columns correspond to time steps
        '''
        C = utils.obj_array(len(Modality))
        C[Modality.WHERE] = np.array([[-1, -1, -1],
                                      [0,   0,  0],
                                      [0,   0,  0],
                                      [0,   0,   0],
                                      [0,   0, 0]])
        C[Modality.WHAT] = np.array([[ 0,  0,  0],
                                     [ c,  c,  c],
                                     [-c, -c, -c]])

        return C

    def create_D(self):
        D = utils.obj_array(len(Modality))
        D[Modality.WHERE] = np.array([1.0, 0.0, 0.0, 0.0])
        D[Modality.WHAT] = norm(np.array([1.0, 1.0]))
        return D

    def create_B_location(self, detail=True,n_policies=4):
        '''
        Used to create the array of probabilities for move
        Row - to state, Col - from state

        Parameters:
            detail     If set to false, create array of allowable moves (ignoring state)
            n_policies Number of policies
        '''
        B = np.zeros((len(Location), len(Location), n_policies))
        B[:, :, 0] = np.array([[1, 1, 0, 0],
                               [0, 0, 0, 0],
                               [0, 0, 1, 0],
                               [0, 0, 0, 1]],
                              dtype=float)
        B[:, :, 1] = np.array([[0, 0, 0, 0],
                               [1, 1, 0, 0],
                               [0, 0, 1, 0],
                               [0, 0, 0, 1]],
                              dtype=float)
        B[:, :, 2] = np.array([[0, 0, 0, 0],
                               [0, 0, 0, 0],
                               [1, 1, 1, 0],
                               [0, 0, 0, 1]],
                              dtype=float)
        B[:, :, 3] = np.array([[0, 0, 0, 0],
                               [0, 0, 0, 0],
                               [0, 0, 1, 0],
                               [1, 1, 0, 1]],
                              dtype=float)
        if detail:
            return B

        return np.sum(B, axis=2)


class Maze(Env):
    '''
    The class represents the generative process for the maze
    '''

    def __init__(self, factory, rng=np.random.default_rng(), p_wrong=2.0 / 100.0):
        '''
        Assign context (left or right) at random, and place mouse at starting position.

        Create a table of allowable transitions by or-ing the matrices from Figure 7.6,
        which  makes Left and Right absorbing states.
        '''
        super().__init__()
        self.rng = rng
        self.location = Location.START
        self.context = Context(self.rng.choice(list(Context)))
        print (f'Context={self.context.name}')
        self.AllowableTransitions = factory.create_B_location(detail=False)
        self.p_wrong = p_wrong

    def reset(self):
        self.location = Location.START

    def step(self, action):
        '''
        Update to position in response to an action
        '''
        if True: #self.AllowableTransitions[action,self.location]:
            self.location = Location(action)

        match self.location:
            case Location.START:
                return LocationObservation.AT_START, Stimulus.NONE

            case Location.BOTTOM:
                if self.context == Context.RIGHT_ATTRACTIVE:
                    if self.rng.uniform() < self.p_wrong:
                        return LocationObservation.AT_BOTTOM_LEFT_ATTRACTIVE, Stimulus.NONE
                    else:
                        return LocationObservation.AT_BOTTOM_RIGHT_ATTRACTIVE, Stimulus.NONE
                else:
                    if self.rng.uniform() < self.p_wrong:
                        return LocationObservation.AT_BOTTOM_RIGHT_ATTRACTIVE, Stimulus.NONE
                    else:
                        return LocationObservation.AT_BOTTOM_LEFT_ATTRACTIVE, Stimulus.NONE

            case Location.LEFT:
                if self.context == Context.RIGHT_ATTRACTIVE:
                    return LocationObservation.AT_LEFT, Stimulus.AVERSIVE
                else:
                    return LocationObservation.AT_LEFT, Stimulus.ATTRACTIVE

            case Location.RIGHT:
                if self.context == Context.RIGHT_ATTRACTIVE:
                    return LocationObservation.AT_RIGHT, Stimulus.ATTRACTIVE
                else:
                    return LocationObservation.AT_RIGHT, Stimulus.AVERSIVE

    def can_move_out(self, action):
        '''
        Verify that the B matrix allows at least one transition to some other state.

        Parameters:
            action  Causes move into some state

        Returns:
            True if we can move to some other state than the one determined by `action`
        '''
        possible_moves = self.AllowableTransitions[:,action]  > 0
        return possible_moves.sum() > 1 or not possible_moves[possible_moves]


def parse_args():
    parser = ArgumentParser(__doc__)
    parser.add_argument('--show', default=False, action='store_true', help='Controls whether plot will be displayed')
    parser.add_argument('--figs', default='./figs', help='Location for storing plot files')
    parser.add_argument('--seed', default=None, type=int, help='Initialize random number generator')
    parser.add_argument('-T', '--Tau', default=5, type=int, help='Number of time steps')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    factory = MazeFactory()
    C=factory.create_C()
    _,n_steps = C[0].shape
    mouse = Agent(A=factory.create_A(),
                  B=factory.create_B(),
                  C=C,
                  D=factory.create_D(),
                  policy_len = n_steps,
                  inference_horizon = n_steps)
    maze = Maze(factory, rng=rng)
    maze.reset()
    action = Move.START
    prior = mouse.D.copy()
    for tau in range(args.Tau):
        o = maze.step(action)
        qs = mouse.infer_states(o)
        q_pi, G = mouse.infer_policies()
        if not maze.can_move_out(action):
            print (o,qs)
            break
        next_action = mouse.sample_action()
        action = Move(int(next_action[0]))
        print (o,qs,action)





