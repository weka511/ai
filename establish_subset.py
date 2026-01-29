#!/usr/bin/env python

#   Copyright (C) 2026 Simon Crase

#   This program is free software: you can redistribute it and/or modify
#   it under the terms of the GNU General Public License as published by
#   the Free Software Foundation, either version 3 of the License, or
#   (at your option) any later version.

#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.

#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.

'''
    Extract subsets of MNIST to facilitate replication

    This program generates a list of indices for data points ensuring that there are
    precisely n points from each class.
'''

from argparse import ArgumentParser
from os.path import join
from pathlib import Path
from time import time
import numpy as np
from mnist import MnistDataloader


def parse_args():
    parser = ArgumentParser(__doc__)
    parser.add_argument('--data', default='./data', help='Location for storing data files')
    parser.add_argument('--out', default=Path(__file__).stem, help='Location for storing indices')
    parser.add_argument('--n', default=1000, type=int, help='Number of images for each class')
    parser.add_argument('--seed', default=None, type=int, help='Seed for random number generator')
    return parser.parse_args()

def extract( ytrain, rng = np.random.default_rng(),nclasses = 10,n=1000):
    '''
    Create list of indices
    '''
    classes = np.zeros((n, nclasses), dtype=int)
    class_counts = np.zeros((nclasses), dtype=int)
    for k in rng.permutation(len(ytrain)):
        image_class = ytrain[k]
        i = class_counts[image_class]
        if i < n:
            classes[i, image_class] = k
            class_counts[image_class] += 1
        else:
            if np.min(class_counts) == n: return classes

if __name__ == '__main__':
    start = time()
    args = parse_args()
    mnist_dataloader = MnistDataloader.create(data=args.data)
    (_, ytrain), _ = mnist_dataloader.load_data()
    classes=extract( ytrain, rng=np.random.default_rng(args.seed),n=args.n)
    np.save(join(args.data, args.out), np.reshape(classes,-1))
    elapsed = time() - start
    minutes = int(elapsed / 60)
    seconds = elapsed - 60 * minutes
    print(f'Elapsed Time {minutes} m {seconds:.2f} s')
