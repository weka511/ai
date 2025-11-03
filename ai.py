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
    Common code for my Active Inference project
'''

from os.path import join
from pathlib import Path
from matplotlib.pyplot import figure, show
from matplotlib import rc

class AxisIterator:
    '''
    This class creates subplots as needed. It enables me to creplot data without distraction.
    '''
    def __init__(self, n_rows=2, n_columns=3, figs='figs', title='', show=False, name=Path(__file__).stem, figsize=None):
        self.figsize = (4*n_columns,4*n_rows) if figsize == None else figsize
        self.n_rows = n_rows
        self.n_columns = n_columns
        self.seq = 0
        self.title = title
        self.figs = figs
        self.show = show
        self.name = name

    def __iter__(self):
        return self

    def __next__(self):
        '''
        Used to supply subplots
        '''
        if self.seq < self.n_rows * self.n_columns:
            self.seq += 1
        else:
            warn('Too many subplots')

        return self.fig.add_subplot(self.n_rows, self.n_columns, self.seq)

    def __enter__(self):
        rc('font', **{'family': 'serif',
                      'serif': ['Palatino'],
                      'size': 8})
        rc('text', usetex=True)
        self.fig = figure(figsize=self.figsize)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.fig.suptitle(self.title, fontsize=12)
        self.fig.tight_layout(pad=3, h_pad=4, w_pad=3)
        self.fig.savefig(join(self.figs, self.name))
        if self.show:
            show()
