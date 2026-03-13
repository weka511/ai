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
    Testbed for Gibbs sampling.
    '''
    def __init__(self):
        super().__init__('Testbed for Gibbs sampling','gibbs')

    def _execute(self):
        '''
        Perform gibbs sampling on specified classes
        '''
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
        
        Parameters:
            x      A matrix, each row being one masked image
            m      Number of rows in x
            f
            
        Returns:
            A array of probabilities
        '''
        MI = np.zeros((m,m))
        for j in range(m):
            y = x[j,:]
            X = x.T
            MI[j,j:] = mutual_info_classif(X[:,j:],y)
            MI[j:,j ] = MI[j,j:]
        Unnormalized = f(MI)
        return Unnormalized/Unnormalized.sum(axis=1)[:,None]

    def gibbs(self,X,N=100,P=np.ones((12,12))):
        '''
        Perform Gibbs sampling

        Parameters:
            X       A matrix, each row being one masked image
            N       Number of iterations
            P       Probability mask created by create_probabilities
        '''
        m,_ = X.shape
        self.links = self.build_initial_links(X,m)
        self.lookup_table = self.create_lookup_table(X,self.links,m)
        for i in range(N):
            if i%5 == 0: self.log(f'Iteration {i+1}')
            break_from,break_to,index = Tower(P,self.links).sample()
            self.log(f'Break link from {break_from} to {break_to}')
            ancestors = self.create_ancestors(break_to,self.links)
            potential_links = self.create_potential_links(m,break_from,ancestors)
            _,link_to,_ = Tower(P,potential_links,f = lambda P:P).sample()
            self.log(f'Make link from {break_from} to {link_to}')
            self.links[index,1] = link_to

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

    def  create_ancestors(self,node,links):
        '''
        Find all ancestors, i.e. things that link to a specific node directly or indirectly.
        
        Parameters:
            node
            links
        '''
        def dfs(link_to,Product):
            '''
            The depth-first search does all the heavy lifting.
            '''
            ancestors = links[links[:,1] == link_to,0]
            if len(ancestors) > 0:
                for p in ancestors:
                    if p != link_to:
                        Product.append(p)
                        dfs(p,Product)
        Product = [node]
        dfs(node,Product)
        return Product

    def create_potential_links(self,m,node,cannot_link_to):
        '''
        Create a collection of nodes we coukd potentially link to
        
        Parameters:
            m
            node
            cannot_link_to
        '''
        Product = np.zeros((m,2),dtype=int)
        i1 = 0
        for i in range(m):
            if i not in cannot_link_to:
                Product[i1,0] = node
                Product[i1,1] = i
                i1 += 1
   
        return Product[0:i1,:]

 

class Tower:
    '''
    This class performs tower sampling, as described in
    Werner Krauth: Statistical Mechanics: Algorithms and Computations ISBN: 9780198515364.
    '''
    def __init__(self,P,links,f=lambda P:1/P,rng=np.random.default_rng()):
        '''
        Build an array of cumulative probabilities for tower sampling
        
        Parameters:
            P
            links
            f
            rng
        '''
        m,_ = links.shape
        self.links = links
        self.probabilities = np.zeros((m))
        self.rng = rng
        for i in range(m):
            j = links[i,0]
            k = links[i,1]
            self.probabilities[i] = f(P[j,k]) + (0 if i == 0 else self.probabilities[i-1])
        self.probabilities /= self.probabilities[-1]
            
    def sample(self):
        '''
        Draw one sample
        
        Returns:
            from
            to
            index
        '''
        sample1 = self.rng.uniform(high=self.probabilities[-1])
        i = np.searchsorted(self.probabilities,sample1)
        return self.links[i,0], self.links[i,1],i
  

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
