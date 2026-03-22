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
from node import Node, NodeSet,Tower
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
            x0 = self.x[self.indices[:,i],:]
            x = self.mask.shorten(self.x[self.indices[:,i],:])
            P = self.create_probabilities(x,m)
            self.gibbs(x,N=self.args.N,P=P)
            self.display(list(self.links.generate_runs()),x0,i)
            
    def display(self,runs,x0,iclass):
        fig = figure(figsize=(12,12))
        m = len(runs)
        n = self.args.nimages
        for j in range(m):
            run = runs[j]
            for k in range(min(len(run),n)):
                ax = fig.add_subplot(m,n,n*j+k+1)
                ax.imshow(x0[run[k],:].reshape(28,28), cmap=self.args.cmap)
                ax.axis('off')
        fig.suptitle(f'Gibbs Sampling: Class={iclass} has {m} styles')
        fig.tight_layout(pad=3,h_pad=3,w_pad=3)
        fig.savefig((self.figs_path / (self.args.out+str(iclass))).with_suffix('.png'))           
                           

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
        self.links = NodeSet.build(m,rng=self.rng)
        for i in range(N):
            if i%5 == 0: self.log(f'Iteration {i+1}')
            break_from,break_to,index = Tower(P,self.links).sample()
            self.log(f'Break link from {break_from} to {break_to}',level=Logger.DEBUG)
            potential_links = self.links.candidate_links(break_from)               
            _,link_to,_ = Tower(P,potential_links,f = lambda P:P).sample()
            self.log(f'Make link from {break_from} to {link_to}',level=Logger.DEBUG)
            self.links.break_link(break_from,break_to)
            self.links.link(break_from,link_to)


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
    parser.add_argument('--N', default=100, type=int, help='Number of iterations for Gibbs sampler')
    parser.add_argument('--nimages', default=11, type=int, help='Maximum number of images for each class')
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
