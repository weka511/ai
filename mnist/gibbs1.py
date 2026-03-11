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


class Gibbs(Stage2):
    '''
    Testbed for Gibbs sampling
    '''
    def __init__(self):
        super().__init__('Testbed for Gibbs sampling','gibbs')

    def _execute(self):
        m,_ = self.indices.shape
        for i in self.args.classes:
            self.log(f'Class {i}')
            x = self.mask.shorten(self.x[self.indices[:,i],:])
            P = self.create_probabilities(x,m)
            self.gibbs(x,N=self.args.N,P=P)

    def create_probabilities(self,x,m,f=np.exp):
        '''
        Create a matrix of probabilities, P[i,j] is
        derived from the mutual informations between i and j.
        '''
        MI = np.zeros((m,m))
        for j in range(m):
            y = x[j,:]
            X = x.T
            MI[j,j:] = mutual_info_classif(X[:,j:],y)
            MI[j:,j ] = MI[j,j:]
        Unnormalized = f(MI)
        return Unnormalized/Unnormalized.sum(axis=1)[:,None]

    def gibbs(self,x,N=100,P=np.ones((12,12))):
        '''
        Perform Gibbs sampling

        Parameters:
            x       Training data for the current digit class, masked
            N       Number of iterations
            P       Probability mask created by create_probabilities
        '''
        m,_ = x.shape
        self.links = self.build_initial_links(x,m)
        self.lookup_table = self.create_lookup_table(x,self.links,m)
        for i in range(N):
            if i%5 == 0: self.log(f'Iteration {i+1}')
            tower1 = Tower(P,self.links,m)
            j = tower1.sample()
            predecessors = self.create_predecessors(j,self.links)
            potential_links = self.create_potential_links(m,j,predecessors)
            tower2 = Tower(P,potential_links,m,f = lambda P:P)
            k = tower2.sample()

            if self.can_break_and_make(j,k):
                pos = self.break_link(j)
                if j != k:
                    self.make_link(j,k)

    def build_initial_links(self,x,m):
        '''
        Create random links for the start of the gibbs MCMC
        
        Parameters:
            x       Training data for the current digit class, masked
            m       Number of training examples
            
        Returns: An m x 2 matrix of indices, each linking to itself, or
                 to an index that has already linked to something, e.g.
                [[21 21]
                 [24 24]
                 [10 10]
                 [16 10]
                 [ 5 16]
                 [12 21]
                 [28 16]
                 [15 16]
                 [11 15]
                 [ 8 21]
                 [ 1 21]
                 [23 28]
                 [19 12]
                 etc
        '''
        Product = np.zeros((m,2),dtype=int)
        Product[:,0] = self.rng.permutation(m)
        Product[0,1] = Product[0,0]
        for i in range(1,m):
            Product[i,1] = self.rng.choice(Product[0:i+1,0])

        return Product

  
    def create_lookup_table(self,x,links,m):
        '''
        Create an auxilary table which tells us where element `i' lives
        on first columns of links
        '''
        product = np.zeros((m),dtype=int)
        for i in range(m):
            product[links[i,0]] = i
        return product

    def  create_predecessors(self,j,links):
        links_sorted = np.sort(links)
        m,_ = links.shape
        Product = []
        j1 = j
        found = True
        while found:
            found = False
            for i in range(m):
                if links[i,1] == j1:
                    found = True
                    j1 = links[i,0]
                    Product.append(j1)
                    break
        return Product

    def create_potential_links(self,m,j,predecessors):
        product = np.zeros((m,2),dtype=int)
        i1 = 0
        for i in range(m):
            if i != j and i not in predecessors:
                product[i1,0] = j
                product[i1,1] = i
                i1 += 1
            n = i
        return product[0:n,:]

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
        i = self.lookup_table[j]
        assert self.links[i,0] == j
        self.links[i,1] = k

class Tower:
    '''
    This class performs tower sampling, as described in
    Werner Krauth: Statistical Mechanics: Algorithms and Computations ISBN: 9780198515364.
    '''
    def __init__(self,P,links,m0,f=lambda P:1/P,rng=np.random.default_rng()):
        '''
        Build an array of cumulative probabilities for tower sampling
        '''
        m,_ = links.shape
        self.probabilities = np.zeros((m))
        self.rng = rng

        for i in range(m):
            j = links[i,0]
            k = links[i,1]
            self.probabilities[i] = f(P[j,k]) + (0 if i == 0 else self.probabilities[i-1])
        self.probabilities /= self.probabilities[-1]
            
    def sample(self):
        sample1 = self.rng.uniform(high=self.probabilities[-1])
        return np.searchsorted(self.probabilities,sample1)
  

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

if __name__ == '__main__':
    rc('font', **{'family': 'serif',
                  'serif': ['Palatino'],
                  'size': 8})
    rc('text', usetex=True)

    Command.build([
        Gibbs()
    ])

    Command.execute_one(parse_args(Command.get_names()))
