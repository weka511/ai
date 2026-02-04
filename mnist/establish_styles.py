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
    Establish styles within classes using mutual information
'''
from argparse import ArgumentParser
from os.path import join
from pathlib import Path
from time import time
from matplotlib.pyplot import figure, show
from matplotlib import rc, cm
import numpy as np
from seaborn import lineplot
from mnist import MnistDataloader, create_mask, columnize
from style import StyleList


def parse_args():
    parser = ArgumentParser(__doc__)
    parser.add_argument('--show', default=False, action='store_true', help='Controls whether plot will be displayed')
    parser.add_argument('--figs', default='./figs', help='Location for storing plot files')
    parser.add_argument('--data', default='./data', help='Location for storing data files')
    parser.add_argument('--indices', default='establish_subset.npy', help='Location where index files have been saved')
    parser.add_argument('--nimages', default=float('inf'), type=int, help='Maximum number of images for each class')
    parser.add_argument('--mask', default=None, help='Name of mask file (omit for no mask)')
    parser.add_argument('--size', default=28, type=int, help='Number of row/cols in each image: shape will be will be mxm')
    parser.add_argument('--classes', default=list(range(10)), type=int, nargs='+', help='List of digit classes')
    parser.add_argument('--bins', default=12, type=int, help='Number of bins for histograms')
    parser.add_argument('--threshold', default=0.1, type=float,help='Include image in same style if mutual information exceeds threshold')
    parser.add_argument('--out', default=Path(__file__).stem, help='Location for storing styles')
    return parser.parse_args()


if __name__ == '__main__':
    rc('font', **{'family': 'serif',
                  'serif': ['Palatino'],
                  'size': 8})
    rc('text', usetex=True)
    start = time()
    args = parse_args()
    rng = np.random.default_rng()
    indices = np.load(join(args.data, args.indices)).astype(int)
    n_examples, n_classes = indices.shape

    mnist_dataloader = MnistDataloader.create(data=args.data)
    (x_train, _), _ = mnist_dataloader.load_data()
    x = columnize(x_train)

    mask, mask_text = create_mask(mask_file=args.mask, data=args.data, size=args.size)
    mask = mask.reshape(-1)
    x = np.multiply(x, mask)
    m, n = indices.shape  # images,classes

    assert n == 10

    for i_class in args.classes:
        style_list = StyleList.build(x, indices,
                                     i_class=i_class,
                                     nimages=min(m,args.nimages),
                                     threshold=args.threshold)
        print(f'Styles for Class {i_class} has length= {len(style_list)}')
        fig = figure(figsize=(8, 8))
        ax1 = fig.add_subplot(1, 1, 1)
        ax1.hist([len(style) for style in style_list.styles])
        ax1.set_title(f'Lengths of style for {len(style_list)} styles')
        fig.suptitle(f'Digit Class = {i_class}, threshold={args.threshold}')
        fig.savefig(join(args.figs, Path(__file__).stem + str(i_class)))
        file = Path(join(args.data, args.out+str(i_class))).with_suffix('.npy')
        style_list.save(file)

    elapsed = time() - start
    minutes = int(elapsed / 60)
    seconds = elapsed - 60 * minutes
    print(f'Elapsed Time {minutes} m {seconds:.2f} s')

    if args.show:
        show()
