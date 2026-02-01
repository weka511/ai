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
    Exploratory Data Analysis for MNIST: figure out variability of
    mutual information within and between classes
'''
from argparse import ArgumentParser
from os.path import join
from pathlib import Path
from time import time
from matplotlib.pyplot import figure, show
from matplotlib import rc, cm
import numpy as np
from sklearn.feature_selection import mutual_info_classif
from mnist import MnistDataloader, create_mask, columnize

def parse_args():
    parser = ArgumentParser(__doc__)
    parser.add_argument('--show', default=False, action='store_true', help='Controls whether plot will be displayed')
    parser.add_argument('--figs', default='./figs', help='Location for storing plot files')
    parser.add_argument('--data', default='./data', help='Location for storing data files')
    parser.add_argument('--indices', default='establish_subset.npy', help='Location for storing data files')
    parser.add_argument('--m', default=12, type=int, help='Number of images for each class')
    parser.add_argument('--mask',default=None,help='Name of mask file (omit for no mask)')
    parser.add_argument('--size', default=28, type=int, help='Number of row/cols in each image: shape will be will be mxm')
    return parser.parse_args()



def create_exemplars(indices,x):
    '''
    Create an array containing one element from each class

    Parameters:
        indices
        x
    '''
    exemplar_indices = indices[0,:]
    return np.array( [ x[i,:] for i in exemplar_indices])

def create_companions(iclass,indices,x,n_comparison=7):
    '''
    Create a collection of vectors belonging to same class

    Parameters:
        iclass
        indices
        x
        n_comparison
    '''
    companion_indices = indices[1:n_comparison+1,iclass]
    return np.array( [ x[i,:] for i in companion_indices])

def annotate(MI,ax=None,color='k'):
    '''
    Annotate heatmap with values of mutual information

    Parameters:
        MI
        ax
        color
    '''
    m,n = MI.shape
    for i in range(m):
        for j in range(n):
            ax.text(j, i, f'{MI[i,j]:.2e}',ha='center', va='center', color='k')

if __name__ == '__main__':
    rc('font', **{'family': 'serif',
                  'serif': ['Palatino'],
                  'size': 8})
    rc('text', usetex=True)
    fig = figure(figsize=(20, 8))
    start = time()
    args = parse_args()
    indices = np.load(join(args.data,args.indices)).astype(int)
    n_examples,n_classes = indices.shape

    mnist_dataloader = MnistDataloader.create(data=args.data)
    (x_train, _), _ = mnist_dataloader.load_data()
    x = columnize(x_train)

    mask,_ = create_mask(mask_file=args.mask,data=args.data,size=args.size)
    mask = mask.reshape(-1)
    x = np.multiply(x,mask)
    Exemplars = create_exemplars(indices,x)
    MI_between_classes = np.zeros((n_classes,n_classes))
    for i in range(n_classes):
        MI_between_classes[i] = mutual_info_classif(Exemplars.T,Exemplars[i,:])

    MI_within_classes = np.zeros((n_classes,args.m))
    for i in range(n_classes):
        companions = create_companions(i,indices,x,n_comparison=args.m)
        MI_within_classes[i] = mutual_info_classif(companions.T,Exemplars[i,:])

    ax1 = fig.add_subplot(1,2,1)
    fig.colorbar(ax1.imshow(MI_between_classes, cmap='Blues', interpolation='nearest'),
                 orientation='vertical')
    ax1.set_title('Mutual Information between classes')
    annotate(MI_between_classes,ax=ax1)

    ax2 = fig.add_subplot(1,2,2)
    fig.colorbar(ax2.imshow(MI_within_classes.T, cmap='Reds', interpolation='nearest'),
                 orientation='vertical')
    ax2.set_title('Mutual Information within classes')
    annotate(MI_within_classes.T,ax=ax2)

    fig.tight_layout(pad=2,h_pad=2,w_pad=2)
    fig.savefig(join(args.figs,Path(__file__).stem))

    elapsed = time() - start
    minutes = int(elapsed/60)
    seconds = elapsed - 60*minutes
    print (f'Elapsed Time {minutes} m {seconds:.2f} s')

    if args.show:
        show()
