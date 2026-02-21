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
Visualise MNIST data
'''
from argparse import ArgumentParser
from os.path import join
from pathlib import Path
from time import time
import numpy as np
from matplotlib.pyplot import figure, show
from matplotlib import rc, cm
from skimage.exposure import equalize_hist
from mnist import MnistDataloader

def parse_args():
    parser = ArgumentParser(__doc__)
    parser.add_argument('--show', default=False, action='store_true', help='Controls whether plot will be displayed')
    parser.add_argument('--figs', default='./figs', help='Location for storing plot files')
    parser.add_argument('--cmap',default='Blues',help='Colour map')
    parser.add_argument('--resize', default=None,type=int,nargs='+',help='Used to resize image')
    parser.add_argument('--m', '-m', default=12,type=int,help='Number of rows for display')
    parser.add_argument('--seed', default=None, type=int, help='For initializing random number generator')
    return parser.parse_args()

if __name__ == '__main__':

    rc('font', **{'family': 'serif',
                  'serif': ['Palatino'],
                  'size': 8})
    rc('text', usetex=True)
    fig = figure(figsize=(8, 12))
    start = time()
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    mnist_dataloader = MnistDataloader.create()
    (x_train,y_train),_ = mnist_dataloader.load_data()

    for i in range(args.m):
        k = rng.choice(len(x_train))
        img = np.array(x_train[k])
        if args.resize != None:
            rows = args.resize[0]
            cols = args.resize[1] if len(args.resize) > 1 else rows
            img = resize(img,(rows,cols))
        ax1 = fig.add_subplot(args.m, 4, 4*i+1)
        ax1.axis('off')
        ax1.imshow(img, cmap=args.cmap)

        ax2 = fig.add_subplot(args.m, 4, 4*i+2)
        ax2.hist(img.reshape(-1),bins=16)
        img_eq = equalize_hist(img)
        ax3 = fig.add_subplot(args.m, 4, 4*i+3)
        ax3.imshow(img_eq, cmap=args.cmap)
        ax3.axis('off')

        ax4 = fig.add_subplot(args.m, 4, 4*i+4)
        ax4.hist(img_eq.reshape(-1))

        if i==0:
            ax2.set_title('Raw' if args.resize == None else 'Resized')
            ax4.set_title('After equalization')

    fig.suptitle('MNIST')
    fig.tight_layout(pad=3,h_pad=3,w_pad=3)
    fig.savefig(join(args.figs,Path(__file__).stem))

    elapsed = time() - start
    minutes = int(elapsed/60)
    seconds = elapsed - 60*minutes
    print (f'Elapsed Time {minutes} m {seconds:.2f} s')

    if args.show:
        show()
