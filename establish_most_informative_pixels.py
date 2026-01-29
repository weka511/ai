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
    Determine which pixels are most relevant to classifying images
'''


from argparse import ArgumentParser
from os.path import splitext,join
from pathlib import Path
from time import time
from matplotlib.pyplot import figure, show
from matplotlib import rc
import numpy as np
from scipy.stats import entropy
from pymdp.maths import softmax, spm_log_single as log_stable
from skimage.exposure import equalize_hist
from skimage.transform import resize
from mnist import MnistDataloader

def parse_args():
    parser = ArgumentParser(__doc__)
    parser.add_argument('--show', default=False, action='store_true', help='Controls whether plot will be displayed')
    parser.add_argument('--figs', default='./figs', help='Location for storing plot files')
    parser.add_argument('--data', default='./data', help='Location for storing data files')
    parser.add_argument('--indices', default='establish_subset.npy', help='Location for storing data files')
    return parser.parse_args()

def create_entropies(x_train,selector,bins=11,m=32):
    n = len(selector)
    m2 = m*m
    def create_1d_images():
        product = np.zeros((n,m2))
        for i in selector:
            img = equalize_hist(resize(np.array(x_train[i]),(m,m)))
            product[i] = np.reshape(img,-1)
        return product

    def create_entropies_from_1d_images(images1d):
        product = np.zeros((m2))
        for i in range((m2)):
            hist,edges = np.histogram(images1d[i],bins=bins,density=True)
            pdf = hist/np.sum(hist)
            product[i] = entropy(pdf)
        return product

    return create_entropies_from_1d_images(create_1d_images())


if __name__ == '__main__':
    rc('font', **{'family': 'serif',
                  'serif': ['Palatino'],
                  'size': 8})
    rc('text', usetex=True)
    fig = figure(figsize=(24, 12))
    start = time()
    args = parse_args()
    indices = np.load(join(args.data,args.indices)).astype(int)

    mnist_dataloader = MnistDataloader.create(data=args.data)
    (x_train, _), _ = mnist_dataloader.load_data()
    x_train = np.array(x_train)
    entropies = create_entropies(x_train[indices],list(range(len(indices))))
    ax = fig.add_subplot(1,1,1)
    fig.colorbar(
        ax.imshow(np.reshape(entropies,(32,32)),
                  cmap='viridis'),
        label='Entropy')

    fig.savefig(join(args.figs,Path(__file__).stem))
    elapsed = time() - start
    minutes = int(elapsed/60)
    seconds = elapsed - 60*minutes
    print (f'Elapsed Time {minutes} m {seconds:.2f} s')

    if args.show:
        show()
