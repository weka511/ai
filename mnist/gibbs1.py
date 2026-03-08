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
    parser.add_argument('--N', default=100, type=int, help='Number of itetaions for Gibbs sampler')
    return parser.parse_args()

class Gibbs(Stage2):
    '''
        Testbed for Gibbs sampling
    '''
    def __init__(self):
        super().__init__('Testbed for Gibbs sampling','gibbs')

    def _execute(self):
        for i in self.args.classes:
            self.log(f'Class {i}')
            self.gibbs(self.mask.shorten(self.x[self.indices[:,i],:]),N=self.args.N)

    def gibbs(self,x,N=100):
        m,_ = x.shape
        self.links = self.build_initial_links(x)
        self.lookup_table = self.create_lookup_table(x,self.links)
        for i in range(N):
            j = self.rng.choice(m)
            k = (j + self.rng.choice(m)) % m
            if self.can_break_and_make(j,k):
                pos = self.break_link(j)
                if j != k:
                    self.make_link(j,k)

    def build_initial_links(self,x):
        m,_ = x.shape
        product = np.zeros((m,2),dtype=int)
        product[:,0] = self.rng.permutation(m)
        product[0,1] = product[0,0]
        for i in range(1,m):
            index_next_customer = self.rng.choice(min(i,m-1))
            product[i,1] = product[index_next_customer,0]

        self.verify_postcondition(product)
        return product

    def verify_postcondition(self,links):
        '''
        Ensure that every item has a link to itself, or to a node that has previously been linked to.
        '''
        m,_ = links.shape
        start = np.sort(links[:,0])
        for i in range(m):
            assert i == start[i]
        destinations = []
        for i in range(m):
            if links[i,0] != links[i,1]:
                assert(links[i,1] in destinations)
            destinations.append(links[i,0])

    def create_lookup_table(self,x,links):
        '''
        Create an auxilary table which tells us where element `i' lives
        on first columns of links
        '''
        m,_ = x.shape
        product = np.zeros((m),dtype=int)
        for i in range(m):
            product[links[i,0]] = i
        return product

    def can_break_and_make(self,j,k):
        print (f'Break {j} and link to {k}')
        i = self.lookup_table[j]
        if self.links[i,1] == j:
            return False
        return True

    def break_link(self,j):
        i = self.lookup_table[j]
        assert self.links[i,0] == j
        self.links[i,1] = j
        return i

    def make_link(self,j,k):
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
