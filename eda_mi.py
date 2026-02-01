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

def columnize(x):
    x1 = np.array(x)
    N,n_rows,n_cols=x1.shape
    assert n_rows == n_cols
    x_img_no_last = np.transpose(x1,[1,2,0])
    x_columnized_img_no_last = np.reshape(x_img_no_last, (n_rows*n_cols, -1))
    return np.transpose(x_columnized_img_no_last,[1,0])

def create_exemplars(indices,x,n_classes=10):
    exemplar_indices = indices[0,:]
    Exemplars = [ x[i,:] for i in exemplar_indices]
    Product = np.zeros((n_classes,28*28))
    for i in range(n_classes):
        for j in range(28*28):
            Product[i,j] = Exemplars[i][j]
    return Product

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

    mask = create_mask(mask_file=args.mask,data=args.data,size=args.size).reshape(-1)

    Exemplars = create_exemplars(indices,x,n_classes=n_classes)
    MI = np.zeros((n_classes,n_classes))
    for i in range(n_classes):
        MI[i] = mutual_info_classif(Exemplars.T,Exemplars[i,:])
    ax = fig.add_subplot(1,2,1)
    heatmap_img = ax.imshow(MI, cmap='Blues', interpolation='nearest')
    fig.colorbar(heatmap_img, orientation='vertical')

    fig.tight_layout(pad=2,h_pad=2,w_pad=2)
    fig.savefig(join(args.figs,Path(__file__).stem))

    if args.show:
        show()
