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
from sklearn.feature_selection import mutual_info_classif
from mnist import MnistDataloader, create_mask,columnize

class Style(object):
    '''
    This class assigns images to Styles
    '''
    def __init__(self,exemplar_index):
        self.exemplar_index = exemplar_index
        self.indices = [exemplar_index]

    def __len__(self):
        return len(self.indices)

    def add(self,new_index):
        self.indices.append(new_index)

class StyleList(object):

    @staticmethod
    def build(x,i_class,n=10,threshold=0.1):
        x_class = x[indices[:,i_class],:]   # All vectors in this digit-class
        style_list = StyleList(x_class)
        for j in range(n):
            matching_style,mi = style_list.get_best_match(j)
            if matching_style == None or mi < args.threshold:
                style_list.add(Style(j))
            else:
                matching_style.add(j)
        return style_list

    def  __init__(self,x_class):
        self.styles = []
        self.x_class = x_class

    def __len__(self):
        return len(self.styles)

    def add(self,style):
        self.styles.append(style)

    def get_best_match(self,index):
        matching_style = None
        mi = 0
        X = self.x_class[index,:].reshape(-1,1)
        for i in range(len(self.styles)):
            candidate_style = self.styles[i]
            y = self.x_class[candidate_style.exemplar_index]
            mi_canditate = mutual_info_classif(X,y)
            if mi_canditate > mi:
                mi = mi_canditate
                matching_style = candidate_style
        return  matching_style,mi

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
    parser.add_argument('--threshold', default = 0.1,type=float)
    return parser.parse_args()

if __name__ == '__main__':
    rc('font', **{'family': 'serif',
                  'serif': ['Palatino'],
                  'size': 8})
    rc('text', usetex=True)
    start = time()
    args = parse_args()
    rng = np.random.default_rng()
    indices = np.load(join(args.data,args.indices)).astype(int)
    n_examples,n_classes = indices.shape

    mnist_dataloader = MnistDataloader.create(data=args.data)
    (x_train, _), _ = mnist_dataloader.load_data()
    x = columnize(x_train)

    mask,mask_text = create_mask(mask_file=args.mask,data=args.data,size=args.size)
    mask = mask.reshape(-1)
    x = np.multiply(x,mask)
    m,n = indices.shape  # images,classes

    assert n == 10

    for i_class in args.classes:
        print (f'Class {i_class}')
        style_list = StyleList.build(x,i_class,n=10,threshold= args.threshold)
        fig = figure(figsize=(8, 8))
        ax1 = fig.add_subplot(1,1,1)
        ax1.hist([len(style) for style in style_list.styles])
        ax1.set_title(f'Lengths of style for {len(style_list)} styles')
        fig.suptitle(f'Digit Class = {i_class}, threshold={args.threshold}')
        fig.savefig(join(args.figs,Path(__file__).stem + str(i_class)))

    elapsed = time() - start
    minutes = int(elapsed/60)
    seconds = elapsed - 60*minutes
    print (f'Elapsed Time {minutes} m {seconds:.2f} s')

    if args.show:
        show()
