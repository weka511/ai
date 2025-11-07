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
from pymdp.maths import softmax, spm_log_single as log_stable, spm_norm as norm
from ai import AxisIterator

class Context(IntEnum):
    RIGHT_ATTRACTIVE = 0
    LEFT_ATTRACTIVE = 1

class Location(IntEnum):
    START = 0
    BOTTOM = 1
    LEFT = 2
    RIGHT = 3

class ChoiceAction(IntEnum):
    MOVE_START = 0
    MOVE_BOTTOM = 1
    MOVE_LEFT = 2
    MOVE_RIGHT = 3
    @classmethod
    def to_location(cls,action):
        match action:
            case ChoiceAction.MOVE_START:
                return Location.START
            case ChoiceAction.MOVE_BOTTOM:
                return Location.BOTTOM
            case ChoiceAction.MOVE_LEFT:
                return Location.LEFT
            case ChoiceAction.MOVE_RIGHT:
                return Location.RIGHT

class LocationObservation(IntEnum):
    AT_START = 0
    AT_BOTTOM_LEFT_ATTRACTIVE = 1
    AT_BOTTOM_RIGHT_ATTRACTIVE = 2
    AT_LEFT = 3
    AT_RIGHT = 4

class Modality(IntEnum):
    WHERE = 0
    WHAT = 1

class Stimulus(IntEnum):
    NONE = 0
    ATTRACTIVE = 1
    AVERSIVE = 2

class Hint(IntEnum):
    HINT_NONE = 0
    HINT_LEFT = 1
    HINT_RIGHT = 2

class MDP_Factory:
    '''
    This class creates the A, B, C, and D matrices for the maze example
    '''
    def create_A(self, probability_hint_wrong=2.0 / 100.0):
        A = utils.obj_array(len(Modality))
        A[Modality.WHERE] = np.empty((len(LocationObservation), len(Location), len(Context)))
        A[Modality.WHERE][:, :, Context.RIGHT_ATTRACTIVE] = np.array([[1.0, 0.0, 0.0, 0.0],
                                                                      [0.0, 0.0, 0.0, 0.0],
                                                                      [0.0, 1.0, 0.0, 0.0],
                                                                      [0.0, 0.0, 1.0, 0.0],
                                                                      [0.0, 0.0, 0.0, 1.0]])
        A[Modality.WHERE][:, :, Context.LEFT_ATTRACTIVE] = np.array([[1.0, 0.0, 0.0, 0.0],
                                                                     [0.0, 1.0, 0.0, 0.0],
                                                                     [0.0, 0.0, 0.0, 0.0],
                                                                     [0.0, 0.0, 1.0, 0.0],
                                                                     [0.0, 0.0, 0.0, 1.0]])
        A[Modality.WHAT] = np.empty((len(Hint), len(Location), len(Context)))
        A[Modality.WHAT][:, :, Context.RIGHT_ATTRACTIVE] = np.array([[1.0, 1.0, 0.0, 0.0],
                                                                     [0.0, 0.0, probability_hint_wrong, 1.0 - probability_hint_wrong],
                                                                     [0.0, 0.0, 1.0 - probability_hint_wrong, probability_hint_wrong]])
        A[Modality.WHAT][:, :, Context.LEFT_ATTRACTIVE] = np.array([[1.0, 1.0, 0.0, 0.0],
                                                                    [0.0, 0.0, 1.0 - probability_hint_wrong, probability_hint_wrong],
                                                                    [0.0, 0.0, probability_hint_wrong, 1.0 - probability_hint_wrong]])

        return A

    def create_B(self):
        B = utils.obj_array(len(Modality))
        B[Modality.WHERE] = np.zeros((len(Location), len(Location), len(ChoiceAction)))
        B[Modality.WHERE][:, :, 0] = np.array([[1.0, 1.0, 0.0, 0.0],
                                  [0.0, 0.0, 0.0, 0.0],
                                  [0.0, 0.0, 1.0, 0.0],
                                  [0.0, 0.0, 0.0, 1.0]])
        B[Modality.WHERE][:, :, 1] = np.array([[0.0, 0.0, 0.0, 0.0],
                                  [1.0, 1.0, 0.0, 0.0],
                                  [0.0, 0.0, 1.0, 0.0],
                                  [0.0, 0.0, 0.0, 1.0]])
        B[Modality.WHERE][:, :, 2] = np.array([[0.0, 0.0, 0.0, 0.0],
                                  [0.0, 0.0, 0.0, 0.0],
                                  [1.0, 1.0, 1.0, 0.0],
                                  [0.0, 0.0, 0.0, 1.0]])
        B[Modality.WHERE][:, :, 3] = np.array([[0.0, 0.0, 0.0, 0.0],
                                  [0.0, 0.0, 0.0, 0.0],
                                  [0.0, 0.0, 1.0, 0.0],
                                  [1.0, 1.0, 0.0, 1.0]])
        B[Modality.WHAT] = np.zeros((len(Context), len(Context), len(ChoiceAction)))
        B[Modality.WHAT][:, :, 0] = np.eye((len(Context)))
        B[Modality.WHAT][:, :, 1] = np.eye((len(Context)))
        B[Modality.WHAT][:, :, 2] = np.eye((len(Context)))
        B[Modality.WHAT][:, :, 3] = np.eye((len(Context)))
        return B

    def create_C(self):
        C = utils.obj_array(len(Modality))
        C[Modality.WHERE] = softmax(np.c_[[-1.0, 0.0, 0.0, 0.0, 0.0]])
        C[Modality.WHAT] = softmax(np.c_[[0.0, 6.0, -6.0]])
        return C

    def create_D(self):
        D = utils.obj_array(len(Modality))
        D[Modality.WHERE] = np.array([1.0, 0.0, 0.0, 0.0])#c_[[1.0, 0.0, 0.0, 0.0]]
        D[Modality.WHAT] = norm(np.array([1.0, 1.0]))#np.c_[[1.0, 1.0]])
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
        self.location = Location.START
        self.context = self.rng.choice([Context.RIGHT_ATTRACTIVE,Context.LEFT_ATTRACTIVE])
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
            self.location = ChoiceAction.to_location(action)

        match self.location:
            case Location.START:
                return LocationObservation.AT_START,Stimulus.NONE

            case Location.BOTTOM:
                if self.context == Context.RIGHT_ATTRACTIVE:
                    if self.rng.uniform() < self.probability_hint_wrong:
                        return LocationObservation.AT_BOTTOM_LEFT_ATTRACTIVE,Stimulus.NONE
                    else:
                        return LocationObservation.AT_BOTTOM_RIGHT_ATTRACTIVE,Stimulus.NONE
                else:
                    if self.rng.uniform() < self.probability_hint_wrong:
                        return LocationObservation.AT_BOTTOM_RIGHT_ATTRACTIVE,Stimulus.NONE
                    else:
                        return LocationObservation.AT_BOTTOM_LEFT_ATTRACTIVE,Stimulus.NONE

            case Location.LEFT:
                if self.context == Context.RIGHT_ATTRACTIVE:
                    return LocationObservation.AT_LEFT,Stimulus.AVERSIVE
                else:
                    return LocationObservation.AT_LEFT,Stimulus.ATTRACTIVE

            case Location.RIGHT:
                if self.context == Context.RIGHT_ATTRACTIVE:
                    return LocationObservation.AT_RIGHT,Stimulus.ATTRACTIVE
                else:
                    return LocationObservation.AT_RIGHT,Stimulus.AVERSIVE


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
    mouse = Agent(A=factory.create_A(), B=factory.create_B(), C=factory.create_C(), D=factory.create_D())
    maze = MazeEnvironment(factory, rng=rng)
    T = 2
    action = 0
    for t in range(T):
        o = maze.step(action)
        qs = mouse.infer_states(o)
        print (o,qs)


    # with AxisIterator(figs=args.figs, title='Section 7.3: Decision Making and Planning as Inference',
                      # show=args.show, name=Path(__file__).stem) as axes:
        # pass

