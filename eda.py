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
    Exploratory Data Analysis for MNIST
'''


from argparse import ArgumentParser
from os.path import join
from pathlib import Path
from time import time
from matplotlib.pyplot import figure, show
from matplotlib import rc, cm
import numpy as np
from scipy.stats import entropy
from pymdp.maths import softmax, spm_log_single as log_stable
from skimage.exposure import equalize_hist
from skimage.transform import resize
from mnist import MnistDataloader, equalize_hist

def parse_args():
    parser = ArgumentParser(__doc__)
    parser.add_argument('--show', default=False, action='store_true', help='Controls whether plot will be displayed')
    parser.add_argument('--figs', default='./figs', help='Location for storing plot files')
    parser.add_argument('--data', default='./data', help='Location for storing data files')
    parser.add_argument('--indices', default='establish_subset.npy', help='Location for storing data files')
    parser.add_argument('--m', default=20, type=int, help='Number of images for each class')
    return parser.parse_args()

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
    m = min(m,args.m)     # Number of columns
    mnist_dataloader = MnistDataloader.create(data=args.data)
    (x_train, _), _ = mnist_dataloader.load_data()
    x_train = np.array(x_train)
    k = 0
    for i in range(n):
        for j in range(m):
            k += 1
            ax = fig.add_subplot(n,m,k)
            img = x_train[indices[j,i]]
            ax.imshow(equalize_hist(img), cmap=cm.gray)
            ax.axis('off')

    fig.tight_layout(pad=2,h_pad=2,w_pad=2)
    fig.savefig(join(args.figs,Path(__file__).stem))

    if args.show:
        show()
