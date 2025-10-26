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

'''HMM Example from Section 7-2'''


# Standard library imports.

from argparse import ArgumentParser
from os.path import join
from pathlib import Path
from matplotlib.pyplot import figure, show
import numpy as np
from pymdp.maths import softmax, spm_log_single as log_stable

if __name__ == '__main__':
    parser = ArgumentParser(__doc__)
    parser.add_argument('--show', default=False, action='store_true', help='Controls whether plot will be displayed')
    parser.add_argument('--figs', default='./figs', help='Location for storing plot files')
    args = parser.parse_args()

    A = 0.1 * np.array([[7, 1, 1, 1],
                        [1, 7, 1, 1],
                        [1, 1, 7, 1],
                        [1, 1, 1, 7]])

    B = 0.01 * np.array([[1, 1, 1, 97],
                         [97, 1, 1, 1],
                         [1, 97, 1, 1],
                         [1, 1, 97, 1]])

    D = np.array([1, 0, 0, 0])

    fig = figure()
    ax = fig.add_subplot(1, 1, 1)

    fig.savefig(join(args.figs, Path(__file__).stem))
    if args.show:
        show()

