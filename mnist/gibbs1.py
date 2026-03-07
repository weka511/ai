#!/usr/bin/env python

# Copyright (C) 2026 Simon Crase  simon@greenweaves.nz

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
    Testbed for Gibbs sampling
'''

from argparse import ArgumentParser
from pathlib import Path
from time import time
import numpy as np
from matplotlib.pyplot import figure, show
from matplotlib import rc, cm
from skimage.exposure import equalize_hist
from skimage.transform import resize
from sklearn.feature_selection import mutual_info_classif
from mnist import MnistDataloader,MnistException
from pipeline import Command,Stage2
from shared.utils import Logger,create_xkcd_colours

def parse_args(names):
    parser = ArgumentParser(__doc__)
    parser.add_argument('command',choices=names,help='The command to be executed')
    parser.add_argument('--show', default=False, action='store_true', help='Controls whether plot will be displayed')
    parser.add_argument('--figs', default='./figs', help='Location for storing plot files')
    parser.add_argument('--data', default='./data', help='Location for storing data files')
    parser.add_argument('--cmap',default='Blues',help='Colour map')
    parser.add_argument('--seed', default=None, type=int, help='For initializing random number generator')
    parser.add_argument('--mask', default=None, help='Name of mask file (omit for no mask)')
    parser.add_argument('--indices', default=None, help='Location where index files have been saved')
    parser.add_argument('--classes', default=list(range(10)), type=int, nargs='+', help='List of digit classes')
    parser.add_argument('-o','--out',nargs='?')
    parser.add_argument('--logs', default='./logs', help='Location for storing log files')
    parser.add_argument('--size', default=28, type=int, help='Number of row/cols in each image: shape will be will be mxm')
    return parser.parse_args()

class Gibbs(Stage2):
    '''
        Testbed for Gibbs sampling
    '''

    def __init__(self):
        super().__init__('Testbed for Gibbs sampling','gibbs',
                         needs_output_file=False)

    def _execute(self):
        pass

if __name__ == '__main__':
    rc('font', **{'family': 'serif',
                  'serif': ['Palatino'],
                  'size': 8})
    rc('text', usetex=True)

    Command.build([
        Gibbs()
    ])

    Command.execute_one(parse_args(Command.get_names()))
