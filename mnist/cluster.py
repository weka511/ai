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
    Cluster Analysis for MNIST: establish variability of
    mutual information within and between classes
'''
from argparse import ArgumentParser
from os.path import join
from pathlib import Path
from time import time
from matplotlib.pyplot import figure, show
from matplotlib import rc, cm
import numpy as np
from seaborn import lineplot
from sklearn.feature_selection import mutual_info_classif
from mnist import MnistDataloader, create_mask,columnize

def parse_args():
    parser = ArgumentParser(__doc__)
    parser.add_argument('--show', default=False, action='store_true', help='Controls whether plot will be displayed')
    parser.add_argument('--figs', default='./figs', help='Location for storing plot files')
    parser.add_argument('--data', default='./data', help='Location for storing data files')
    parser.add_argument('--indices', default='establish_subset.npy', help='Location for storing data files')
    parser.add_argument('--npairs', default=128, type=int, help='Number of pairs for each class')
    parser.add_argument('--mask',default=None,help='Name of mask file (omit for no mask)')
    parser.add_argument('--size', default=28, type=int, help='Number of row/cols in each image: shape will be will be mxm')
    parser.add_argument('--classes',default=list(range(10)),type=int,nargs='+',help='List of digit classes')
    parser.add_argument('--bins', default=12, type=int, help='Number of bins for histograms')
    return parser.parse_args()

def create_frequencies(x,indices,bins=[],npairs=128,m=1000):
    '''
    Generate a histogram of mutual information between pairs of images from the same digit class

    Parameters:
        x         Image data
        npairs    Number of pairs to select
        m         Number of images in class
    '''
    def create_mutual_info():
        product = np.zeros((npairs))
        for i in range(npairs):
            K = rng.choice(m,size=2)
            x_class = x[indices[K,i_class],:]
            mi = mutual_info_classif(x_class.T,x_class[0,:])
            product[i] = mi[-1]
        return product

    return np.histogram(create_mutual_info(),bins,density=True)[0]

if __name__ == '__main__':
    rc('font', **{'family': 'serif',
                  'serif': ['Palatino'],
                  'size': 8})
    rc('text', usetex=True)
    start = time()
    args = parse_args()
    rng = np.random.default_rng()
    fig = figure(figsize=(8, 8))
    indices = np.load(join(args.data,args.indices)).astype(int)
    n_examples,n_classes = indices.shape

    mnist_dataloader = MnistDataloader.create(data=args.data)
    (x_train, _), _ = mnist_dataloader.load_data()
    x = columnize(x_train)

    mask,mask_text = create_mask(mask_file=args.mask,data=args.data,size=args.size)
    mask = mask.reshape(-1)
    x = np.multiply(x,mask)
    m,n = indices.shape  # images,classes
    npairs = min(m,args.npairs)
    bins = np.linspace(0,1,num=args.bins+1)
    assert n == 10
    ax = fig.add_subplot(1,1,1)
    for i_class in args.classes:
        print (f'Class {i_class}')
        ax.plot(0.5*(bins[:-1] + bins[1:]), create_frequencies(x,indices,bins,npairs=npairs,m=m),label=str(i_class))

    ax.set_xlabel('Mutual Information')
    ax.set_ylabel('Frequency')
    ax.set_title(f'Mutual Information within classes based on {npairs} pairs, {mask_text}')
    ax.legend(title='Digit classes')
    fig.savefig(join(args.figs,Path(__file__).stem))

    elapsed = time() - start
    minutes = int(elapsed/60)
    seconds = elapsed - 60*minutes
    print (f'Elapsed Time {minutes} m {seconds:.2f} s')

    if args.show:
        show()
