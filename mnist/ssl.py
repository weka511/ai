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
    Display representatives of all styles created by establish_styles.py
'''
from abc import ABC,abstractmethod
from argparse import ArgumentParser
from os.path import join
from pathlib import Path
from time import time, strftime,localtime
from matplotlib.pyplot import figure, show
from matplotlib import rc, cm
import numpy as np
from mnist import MnistDataloader, create_mask, columnize
from style import StyleList

class Command(ABC):
    '''
    Parent class for procesing requests
    '''
    commands = {}

    @staticmethod
    def build(command_list):
        for command in command_list:
            Command.commands[command.command_name] = command

    @staticmethod
    def get_command_names():
        return [name for name in Command.commands.keys()]

    def __init__(self,name,command_name):
        self.name = name
        self.command_name = command_name

    def get_name(self):
        return self.name

    def set_args(self,args):
        self.args = args
        self.rng = np.random.default_rng()           #TODO add seed

    def execute(self):
        '''
        Execute command: shared code
        '''
        print (self.get_name(),strftime("%a, %d %b %Y %H:%M:%S +0000", localtime()))
        self._execute()

    @abstractmethod
    def _execute(self):
        '''
        Execute command
        '''
        ...

class DisplayStyles(Command):
    '''
        Display representatives of all styles created by establish_styles.py
    '''
    def __init__(self):
        super().__init__('Display Styles','display-styles')

    def _execute(self):
        indices = np.load(join(self.args.data, self.args.indices)).astype(int) # no need as already selected
        n_examples, n_classes = indices.shape

        mnist_dataloader = MnistDataloader.create(data=self.args.data)
        (x_train, _), _ = mnist_dataloader.load_data()
        x = columnize(x_train)

        mask, mask_text = create_mask(mask_file=self.args.mask, data=self.args.data, size=self.args.size)
        mask = mask.reshape(-1)
        x = np.multiply(x, mask)
        m, n = indices.shape  # images,classes

        assert n == 10

        for i_class in self.args.classes:
            fig = figure(figsize=(8, 8))
            x_class = x[indices[:,i_class],:]
            Allocation = np.load(join(self.args.data, self.args.styles+str(i_class)+'.npy')).astype(int)
            m1,n1 = Allocation.shape
            if self.args.nimages != None:
                n1 = min(n1,self.args.nimages)
            for j in range(m1):
                for k in range(n1):
                    ax = fig.add_subplot(m1, n1, j*n1 + k + 1)
                    img = x_class[Allocation[j,k]].reshape(args.size,args.size)
                    ax.imshow(img,cmap=cm.gray)

class EstablishStyles(Command):
    def __init__(self):
        super().__init__('Dummy','establish-styles')

    def _execute(self):
        print (self.command_name)

def parse_args(command_names):
    parser = ArgumentParser(__doc__)
    parser.add_argument('command',choices=command_names)
    parser.add_argument('--show', default=False, action='store_true', help='Controls whether plot will be displayed')
    parser.add_argument('--figs', default='./figs', help='Location for storing plot files')
    parser.add_argument('--data', default='./data', help='Location for storing data files')
    parser.add_argument('--indices', default='establish_subset.npy', help='Location where index files have been saved')
    parser.add_argument('--nimages', default=None, type=int, help='Maximum number of images for each class')
    parser.add_argument('--mask', default=None, help='Name of mask file (omit for no mask)')
    parser.add_argument('--size', default=28, type=int, help='Number of row/cols in each image: shape will be will be mxm')
    parser.add_argument('--classes', default=list(range(10)), type=int, nargs='+', help='List of digit classes')
    parser.add_argument('--bins', default=12, type=int, help='Number of bins for histograms')
    parser.add_argument('--threshold', default=0.1, type=float,help='Include image in same style if mutual information exceeds threshold')
    parser.add_argument('--styles', default=Path(__file__).stem, help='Location where styles have been stored')
    return parser.parse_args()


if __name__ == '__main__':
    rc('font', **{'family': 'serif',
                  'serif': ['Palatino'],
                  'size': 8})
    rc('text', usetex=True)
    start = time()
    Command.build([
        DisplayStyles(),
        EstablishStyles()
    ])
    args = parse_args(Command.get_command_names())
    command = Command.commands[args.command]
    command.set_args(args)
    command.execute()

    elapsed = time() - start
    minutes = int(elapsed / 60)
    seconds = elapsed - 60 * minutes
    print(f'Elapsed Time {minutes} m {seconds:.2f} s')

    if args.show:
        show()
