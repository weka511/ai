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
from skimage.transform import resize
from mnist import MnistDataloader
from pipeline import Command

class Visualize(Command):
    '''
        Visualise histogram equalization of MNIST data
    '''
    def __init__(self):
        super().__init__(' Visualise histogram equalization of MNIST data','histeq',
                         needs_output_file=False,
                         needs_index_file=False)

    def _execute(self):
        fig = figure(figsize=(8, 12))
        for i in range(args.m):
            k = self.rng.choice(len(self.x_train))
            img = np.array(self.x_train[k])
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

class EDA(Command):
    '''
    Exploratory Data Analysis for MNIST: plot some raw data, with or without masking
    '''
    def __init__(self):
        super().__init__('Exploratory Data Analysis','eda')

    def _execute(self):
        '''
        Plot some raw data, with or without masking
        '''
        fig = figure(figsize=(20, 8))
        for k,img in self.generate_images(classes=args.classes,m=args.images_per_digit,size=args.size):
            ax = fig.add_subplot(len(args.classes),args.images_per_digit,k)
            ax.imshow(img, cmap=args.cmap)
            ax.axis('off')

        fig.suptitle(('No mask' if args.mask == None
                      else rf'Mask preserving {int(100*self.mask.sum()/(self.args.size*self.args.size))}\% of pixels'))

        fig.tight_layout(pad=2,h_pad=2,w_pad=2)
        fig.savefig(join(args.figs,Path(__file__).stem))

    def generate_images(self,classes=list(range(10)),m=20,size=28):
        '''
        Used to iterate through all the images that need to be displayed

        Parameters:
            classes    List of classes to be displayed
            m          Number of images for each class
            size       Size of image size x size
        '''
        x = np.array(self.x_train)
        k = 0
        for i in classes:
            for j in range(m):
                k += 1
                yield k,resize(x[self.indices[j,i]],(size,size))

def parse_args(names):
    parser = ArgumentParser(__doc__)
    parser.add_argument('command',choices=names,help='The command to be executed')
    parser.add_argument('--show', default=False, action='store_true', help='Controls whether plot will be displayed')
    parser.add_argument('--figs', default='./figs', help='Location for storing plot files')
    parser.add_argument('--data', default='./data', help='Location for storing data files')
    parser.add_argument('--cmap',default='Blues',help='Colour map')
    parser.add_argument('--resize', default=None,type=int,nargs='+',help='Used to resize image')
    parser.add_argument('--m', '-m', default=12,type=int,help='Number of rows for display')
    parser.add_argument('--seed', default=None, type=int, help='For initializing random number generator')
    parser.add_argument('--mask', default=None, help='Name of mask file (omit for no mask)')
    parser.add_argument('--size', default=28, type=int, help='Number of row/cols in each image: shape will be will be mxm')
    parser.add_argument('--indices', default=None, help='Location where index files have been saved')
    parser.add_argument('--classes', default=list(range(10)), type=int, nargs='+', help='List of digit classes')

    group_eda = parser.add_argument_group('Options for eda')
    group_eda.add_argument('--images_per_digit',default=8,type=int,help='Number of images in each digit class')

    return parser.parse_args()

if __name__ == '__main__':

    rc('font', **{'family': 'serif',
                  'serif': ['Palatino'],
                  'size': 8})
    rc('text', usetex=True)

    start = time()

    Command.build([Visualize(),EDA()])
    args = parse_args(Command.get_names())
    command = Command.commands[args.command]
    command.set_args(args)
    try:
        command.execute()
    except FileNotFoundError as e:
        print(f'Error: {e.filename} not found.')
        exit (1)

    elapsed = time() - start
    minutes = int(elapsed/60)
    seconds = elapsed - 60*minutes
    print (f'Elapsed Time {minutes} m {seconds:.2f} s')

    if args.show:
        show()
