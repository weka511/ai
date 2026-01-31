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
    Exploratory Data Analysis for MNIST: plot some raw data, with and without masking
'''
from argparse import ArgumentParser
from os.path import join
from pathlib import Path
from time import time
from matplotlib.pyplot import figure, show
from matplotlib import rc, cm
import numpy as np
from scipy.stats import entropy
from skimage.transform import resize
from mnist import MnistDataloader, create_mask

def parse_args():
    parser = ArgumentParser(__doc__)
    parser.add_argument('--show', default=False, action='store_true', help='Controls whether plot will be displayed')
    parser.add_argument('--figs', default='./figs', help='Location for storing plot files')
    parser.add_argument('--data', default='./data', help='Location for storing data files')
    parser.add_argument('--indices', default='establish_subset.npy', help='Location for storing data files')
    parser.add_argument('--m', default=20, type=int, help='Number of images for each class')
    parser.add_argument('--mask',default=None,help='Name of mask file (omit for no mask)')
    parser.add_argument('--size', default=28, type=int, help='Number of row/cols in each image  will be mxm')
    return parser.parse_args()

def generate_images(x,indices,n=10,m=20,size=28):
    '''
    Used to iterate through all the images that need to be displayed

    Parameters:
        x          Data to be plotted
        indices    Indices for selecting data
        n          Number of classes
        m          Number of images for each class
        size       Size of image size x size
    '''
    k = 0
    for i in range(n):
        for j in range(m):
            k += 1
            yield k,resize(x[indices[j,i]],(size,size))

if __name__ == '__main__':
    rc('font', **{'family': 'serif',
                  'serif': ['Palatino'],
                  'size': 8})
    rc('text', usetex=True)
    fig = figure(figsize=(20, 8))
    start = time()
    args = parse_args()
    indices = np.load(join(args.data,args.indices)).astype(int)
    m,n = indices.shape
    m = min(m,args.m)     # Number of columns for showing images
    mnist_dataloader = MnistDataloader.create(data=args.data)
    (x_train, _), _ = mnist_dataloader.load_data()
    x_train = np.array(x_train)
    mask = create_mask(mask_file=args.mask,data=args.data,size=args.size)

    for k,img in generate_images(x_train,indices,n=n,m=m,size=args.size):
        ax = fig.add_subplot(n,m,k)
        ax.imshow(np.multiply(img,mask), cmap=cm.Blues)
        ax.axis('off')

    fig.suptitle(('No mask' if args.mask == None
                  else rf'Mask preserving {int(100*mask.sum()/(args.size*args.size))}\% of pixels'))

    fig.tight_layout(pad=2,h_pad=2,w_pad=2)
    fig.savefig(join(args.figs,Path(__file__).stem))

    if args.show:
        show()
