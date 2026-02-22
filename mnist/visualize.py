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
from sklearn.feature_selection import mutual_info_classif
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
        for i in range(self.args.m):
            k = self.rng.choice(len(self.x_train))
            img = np.array(self.x_train[k])
            if self.args.resize != None:
                rows = self.args.resize[0]
                cols = self.args.resize[1] if len(self.args.resize) > 1 else rows
                img = resize(img,(rows,cols))
            ax1 = fig.add_subplot(self.args.m, 4, 4*i+1)
            ax1.axis('off')
            ax1.imshow(img, cmap=self.args.cmap)

            ax2 = fig.add_subplot(self.args.m, 4, 4*i+2)
            ax2.hist(img.reshape(-1),bins=16)
            img_eq = equalize_hist(img)
            ax3 = fig.add_subplot(self.args.m, 4, 4*i+3)
            ax3.imshow(img_eq, cmap=self.args.cmap)
            ax3.axis('off')

            ax4 = fig.add_subplot(self.args.m, 4, 4*i+4)
            ax4.hist(img_eq.reshape(-1))

            if i==0:
                ax2.set_title('Raw' if self.args.resize == None else 'Resized')
                ax4.set_title('After equalization')

        fig.suptitle('MNIST')
        fig.tight_layout(pad=3,h_pad=3,w_pad=3)
        fig.savefig(join(self.args.figs,Path(__file__).stem))

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
        for k,img in self.generate_images(classes=self.args.classes,m=self.args.images_per_digit,size=self.args.size):
            ax = fig.add_subplot(len(self.args.classes),self.args.images_per_digit,k)
            ax.imshow(img, cmap=self.args.cmap)
            ax.axis('off')

        fig.suptitle(('No mask' if self.args.mask == None
                      else rf'Mask preserving {int(100*self.mask.sum()/(self.args.size*self.args.size))}\% of pixels'))

        fig.tight_layout(pad=2,h_pad=2,w_pad=2)
        fig.savefig(join(self.args.figs,Path(__file__).stem))

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

class EDA_MI(Command):
    '''
    Exploratory Data Analysis for MNIST: determine variability of
    mutual information within and between classes
    '''
    def __init__(self):
        super().__init__('EDA Mutual Information','eda-mi')

    '''
    Determine variability of mutual information within and between classes
    '''
    def _execute(self):
        n_examples,n_classes = self.indices.shape
        Exemplars = self.create_exemplars()
        MI_between_classes = np.zeros((n_classes,n_classes))
        for i in range(n_classes):
            MI_between_classes[i] = mutual_info_classif(Exemplars.T,Exemplars[i,:])

        MI_within_classes = np.zeros((n_classes,self.args.nimages))
        for i in range(n_classes):
            companions = self.create_companions(i,n_comparison=self.args.nimages)
            MI_within_classes[i] = mutual_info_classif(companions.T,Exemplars[i,:])

        fig = figure(figsize=(20, 8))
        ax1 = fig.add_subplot(1,2,1)
        fig.colorbar(ax1.imshow(MI_between_classes, cmap=self.args.cmap, interpolation='nearest'),
                     orientation='vertical')
        ax1.set_title('Mutual Information between classes')
        EDA_MI.annotate(MI_between_classes,ax=ax1)

        ax2 = fig.add_subplot(1,2,2)
        fig.colorbar(ax2.imshow(MI_within_classes.T, cmap='Reds', interpolation='nearest'),
                     orientation='vertical')
        ax2.set_title('Mutual Information within classes')
        EDA_MI.annotate(MI_within_classes.T,ax=ax2)

        fig.tight_layout(pad=2,h_pad=2,w_pad=2)
        fig.savefig(join(self.args.figs,Path(__file__).stem))

    def create_exemplars(self):
        '''
        Create an array containing one element from each class
        '''
        exemplar_indices = self.indices[0,:]
        return np.array( [ self.x[i,:] for i in exemplar_indices])

    def create_companions(self,iclass,n_comparison=7):
        '''
        Create a collection of vectors belonging to same class

        Parameters:
            iclass         The digit class we are considering
            n_comparison   The number of images we will compare
        '''
        companion_indices = self.indices[1:n_comparison+1,iclass]
        return np.array( [ self.x[i,:] for i in companion_indices])

    @staticmethod
    def annotate(MI,ax=None,color='k'):
        '''
        Annotate heatmap with values of mutual information

        Parameters:
            MI          Vaues for annotation
            ax          Axis for display
            color       Colour for annotations
        '''
        m,n = MI.shape
        for i in range(m):
            for j in range(n):
                ax.text(j, i, f'{MI[i,j]:.2e}',ha='center', va='center', color='k')

class Cluster(Command):
    '''
    Plot mutual information between classes
    '''
    def __init__(self):
        super().__init__('Cluster','explore-clusters',needs_output_file=True)

    '''
    Plot mutual information between classes
    '''
    def _execute(self):
        fig = figure(figsize=(8, 8))
        m,n = self.indices.shape
        npairs = min(m,self.args.npairs)
        bins = np.linspace(0,1,num=self.args.bins+1)
        assert n == 10
        ax = fig.add_subplot(1,1,1)
        for i_class in self.args.classes:
            print (f'Class {i_class}')
            ax.plot(0.5*(bins[:-1] + bins[1:]),
                    self.create_frequencies(i_class,bins,npairs=npairs,m=m,rng=self.rng),label=str(i_class))

        ax.set_xlabel('Mutual Information')
        ax.set_ylabel('Frequency')
        ax.set_title(f'Mutual Information within classes based on {npairs} pairs, {self.mask_text}')
        ax.legend(title='Digit classes')
        fig.savefig(join(self.args.figs,self.args.out))

    def create_frequencies(self,i_class,bins=[],npairs=128,m=1000,rng=None):
        '''
        Generate a histogram of mutual information between pairs of images from the same digit class

        Parameters:
            npairs    Number of pairs to select
            m         Number of images in class
        '''
        def create_mutual_info():
            product = np.zeros((npairs))
            for i in range(npairs):
                K = rng.choice(m,size=2)
                x_class = self.x[self.indices[K,i_class],:]
                mi = mutual_info_classif(x_class.T,x_class[0,:])
                product[i] = mi[-1]
            return product

        return np.histogram(create_mutual_info(),bins,density=True)[0]

class DisplayStyles(Command):
    '''
    Display representatives of all styles created by EstablishStyles
    '''
    def __init__(self):
        super().__init__('Display Styles','display-styles',
                         needs_style_file=True)

    def _execute(self):
        '''
        Display representatives of all styles created by establish-styles
        '''
        for i_class in self.args.classes:
            fig = figure(figsize=(8, 8))
            x_class = self.x[self.indices[:,i_class],:]
            n_styles,n_images = self.Allocations[i_class].shape
            if self.args.nimages != None:
                n_images = min(n_images,self.args.nimages)
            n_styles = min(n_styles,self.args.nstyles)
            for j in range(n_styles):
                for k in range(n_images):
                    ax = fig.add_subplot(n_styles, n_images, j*n_images + k + 1)
                    img = x_class[self.Allocations[i_class][j,k]].reshape(self.args.size,self.args.size)
                    ax.imshow(img,cmap=self.args.cmap)
                    ax.axis('off')
            fig.tight_layout(pad=2,h_pad=2,w_pad=2)
            fig.savefig(Path(join(self.args.figs, self.args.styles+str(i_class))).with_suffix('.png'))


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
    parser.add_argument('--nimages', default=11, type=int, help='Maximum number of images for each class')
    parser.add_argument('-o','--out',nargs='?')
    parser.add_argument('--bins', default=12, type=int, help='Number of bins for histograms')

    group_eda = parser.add_argument_group('Options for eda')
    group_eda.add_argument('--images_per_digit',default=8,type=int,help='Number of images in each digit class')

    group_explore_clusters = parser.add_argument_group('Options for explore-clusters')
    group_explore_clusters.add_argument('--npairs', default=128, type=int, help='Number of pairs for each class')

    group_display_styles = parser.add_argument_group('Options for display-styles')
    group_display_styles.add_argument('--styles', default=Path(__file__).stem, help='Location where styles have been stored')
    group_display_styles.add_argument('--nstyles', default=7, type=int,help='Maximum number of styles to be displayed')


    return parser.parse_args()

if __name__ == '__main__':

    rc('font', **{'family': 'serif',
                  'serif': ['Palatino'],
                  'size': 8})
    rc('text', usetex=True)

    start = time()

    Command.build([
        Visualize(),
        Cluster(),
        DisplayStyles(),
        EDA(),
        EDA_MI()
    ])

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
