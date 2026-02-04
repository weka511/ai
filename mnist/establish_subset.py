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

    This program generates a list of indices for data points ensuring
    that there are a specified number of images from each class.
'''

from argparse import ArgumentParser
from os.path import join
from pathlib import Path
from time import time
import numpy as np
from mnist import MnistDataloader,create_indices

def parse_args():
    parser = ArgumentParser(__doc__)
    parser.add_argument('--data', default='./data', help='Location for storing data files')
    parser.add_argument('--out', default=Path(__file__).stem, help='Location for storing indices')
    parser.add_argument('--nimages', default=1000, type=int, help='Number of images for each class')
    parser.add_argument('--seed', default=None, type=int, help='Seed for random number generator')
    return parser.parse_args()

if __name__ == '__main__':
    start = time()
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    mnist_dataloader = MnistDataloader.create(data=args.data)
    (_, y), _ = mnist_dataloader.load_data()
    indices = create_indices(y, nimages=args.nimages, rng=rng)
    m, n = indices.shape
    file = Path(join(args.data, args.out)).with_suffix('.npy')
    np.save(file, indices)
    print(f'Saved {m} labels for each of {n} classes in {file.resolve()}')

    elapsed = time() - start
    minutes = int(elapsed / 60)
    seconds = elapsed - 60 * minutes
    print(f'Elapsed Time {minutes} m {seconds:.2f} s')
