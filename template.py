#!/usr/bin/env python

# Copyright (C) 2023 Simon Crase  simon@greenweaves.nz

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

'''Template for Python code'''


# Standard library imports.

from argparse import ArgumentParser
from os.path  import join
from pathlib  import Path

# Related third party imports.

from matplotlib.pyplot import figure, show
import numpy as np

# Local application/library specific imports.


if __name__=='__main__':
    parser = ArgumentParser(__doc__)
    parser.add_argument('--show', default=False, action='store_true', help='Controls whether plot will be displayed')
    parser.add_argument('--figs', default='./figs',                   help = 'Location for storing plot files')
    args = parser.parse_args()

    fig = figure()
    ax  = fig.add_subplot(1,1,1)

    fig.savefig(join(args.figs,Path(__file__).stem))
    if args.show:
        show()

