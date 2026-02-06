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
    Display representatives of all styles created by establish_styles.py
'''
from abc import ABC,abstractmethod
from argparse import ArgumentParser
from os.path import join
from pathlib import Path
from time import time, strftime,localtime
from matplotlib.pyplot import figure, show
from matplotlib import rc, cm
import numpy as np
from mnist import MnistDataloader, create_mask, columnize,create_indices
from scipy.stats import entropy
from skimage.exposure import equalize_hist
from sklearn.feature_selection import mutual_info_classif
from skimage.transform import resize
from style import StyleList

class Command(ABC):
    '''
    Parent class for procesing requests
    '''
    commands = {}

    @staticmethod
    def build(command_list):
        for command in command_list:
            Command.commands[command.command_name] = command

    @staticmethod
    def get_command_names():
        return [name for name in Command.commands.keys()]

    def __init__(self,name,command_name,needs_output_file=False):
        self.name = name
        self.command_name = command_name
        self.needs_output_file = needs_output_file

    def get_name(self):
        return self.name

    def set_args(self,args):
        self.args = args
        self.rng = np.random.default_rng()           #TODO add seed

    def execute(self):
        '''
        Execute command: shared code
        '''
        print (self.get_name(),strftime("%a, %d %b %Y %H:%M:%S +0000", localtime()))
        if self.needs_output_file and args.out == None:
            print ('Output file must be specified')
            exit(1)
        mnist_dataloader = MnistDataloader.create(data=self.args.data)
        (self.x_train, self.ytrain), _ = mnist_dataloader.load_data()
        x = columnize(self.x_train)

        mask, self.mask_text = create_mask(mask_file=self.args.mask, data=self.args.data, size=self.args.size)
        mask = mask.reshape(-1)
        self.x = np.multiply(x, mask)
        self.indices = np.load(join(self.args.data, self.args.indices)).astype(int)
        n_examples, n_classes = self.indices.shape
        assert n_classes == 10
        self._execute()

    @abstractmethod
    def _execute(self):
        '''
        Execute command
        '''
        ...

class EDA(Command):
    '''
        Exploratory Data Analysis for MNIST: plot some raw data, with and without masking
    '''
    def __init__(self):
        super().__init__('Exploratory Data Analysis','eda')

    def _execute(self):
        m,n = 10,10  #FIXME
        fig = figure(figsize=(20, 8))
        for k,img in self.generate_images(n=n,m=args.nimages,size=args.size):
            ax = fig.add_subplot(n,args.nimages,k)
            ax.imshow(img, cmap=cm.Blues)
            ax.axis('off')

        fig.suptitle(('No mask' if args.mask == None
                      else rf'Mask preserving {int(100*mask.sum()/(args.size*args.size))}\% of pixels'))

        fig.tight_layout(pad=2,h_pad=2,w_pad=2)
        fig.savefig(join(args.figs,Path(__file__).stem))

    def generate_images(self,n=10,m=20,size=28):
        '''
        Used to iterate through all the images that need to be displayed

        Parameters:
            x          Data to be plotted
            indices    Indices for selecting data
            n          Number of classes
            m          Number of images for each class
            size       Size of image size x size
        '''
        x = np.array(self.x_train)
        k = 0
        for i in range(n):
            for j in range(m):
                k += 1
                yield k,resize(x[self.indices[j,i]],(size,size))


class EDA_MI(Command):
    '''
    Exploratory Data Analysis for MNIST: figure out variability of
    mutual information within and between classes
    '''
    def __init__(self):
        super().__init__('EDA Mutual Information','eda-mi')

    def _execute(self):
        n_examples,n_classes = self.indices.shape
        Exemplars = self.create_exemplars()
        MI_between_classes = np.zeros((n_classes,n_classes))
        for i in range(n_classes):
            MI_between_classes[i] = mutual_info_classif(Exemplars.T,Exemplars[i,:])

        MI_within_classes = np.zeros((n_classes,args.m))
        for i in range(n_classes):
            companions = self.create_companions(i,n_comparison=args.m)
            MI_within_classes[i] = mutual_info_classif(companions.T,Exemplars[i,:])
        fig = figure(figsize=(20, 8))
        ax1 = fig.add_subplot(1,2,1)
        fig.colorbar(ax1.imshow(MI_between_classes, cmap='Blues', interpolation='nearest'),
                     orientation='vertical')
        ax1.set_title('Mutual Information between classes')
        EDA_MI.annotate(MI_between_classes,ax=ax1)

        ax2 = fig.add_subplot(1,2,2)
        fig.colorbar(ax2.imshow(MI_within_classes.T, cmap='Reds', interpolation='nearest'),
                     orientation='vertical')
        ax2.set_title('Mutual Information within classes')
        EDA_MI.annotate(MI_within_classes.T,ax=ax2)

        fig.tight_layout(pad=2,h_pad=2,w_pad=2)
        fig.savefig(join(args.figs,Path(__file__).stem))

    def create_exemplars(self):
        '''
        Create an array containing one element from each class

        Parameters:
            indices
            x
        '''
        exemplar_indices = self.indices[0,:]
        return np.array( [ self.x[i,:] for i in exemplar_indices])

    def create_companions(self,iclass,n_comparison=7):
        '''
        Create a collection of vectors belonging to same class

        Parameters:
            iclass
            indices
            x
            n_comparison
        '''
        companion_indices = self.indices[1:n_comparison+1,iclass]
        return np.array( [ self.x[i,:] for i in companion_indices])

    @staticmethod
    def annotate(MI,ax=None,color='k'):
        '''
        Annotate heatmap with values of mutual information

        Parameters:
            MI
            ax
            color
        '''
        m,n = MI.shape
        for i in range(m):
            for j in range(n):
                ax.text(j, i, f'{MI[i,j]:.2e}',ha='center', va='center', color='k')


class EstablishPixels(Command):
    '''
        Determine which pixels are most relevant to classifying images
    '''
    def __init__(self):
        super().__init__('Establish Pixels','establish-pixels',needs_output_file=True)

    def _execute(self):
        x_train = np.array(self.x_train)
        indices = self.indices.reshape(-1)
        entropies = self.create_entropies(x_train[indices],list(range(len(indices))),bins=args.bins,m=args.size) # Issue 30
        mu = np.mean(entropies)
        sigma = np.std(entropies)
        min0 = np.min(entropies)
        img = np.reshape(entropies,(args.size,args.size)) # Issue 30
        mask = self.cull(entropies,-args.threshold,mu,sigma,clip=True).reshape(args.size,args.size) # Issue 30
        file = Path(join(args.data, args.out)).with_suffix('.npy')
        np.save(file, mask)
        fig = figure(figsize=(12, 12))
        self.show_image(img,ax=fig.add_subplot(2,2,1),fig=fig,cmap=args.cmap)
        self.show_culled(img,-args.threshold,mu,sigma,min0,ax = fig.add_subplot(2,2,2),cmap=args.cmap)
        self.show_mask(mask,cmap=args.cmap,ax = fig.add_subplot(2,2,3),size=args.size)
        self.show_histogram(img,mu,sigma,threshold=args.threshold,ax=fig.add_subplot(2,2,4))

        fig.suptitle(r'Processed \emph{' + f'{args.indices}'
                     ',} '  f'{len(indices) // 10:,d} images per class, {args.bins} bins')
        fig.savefig(join(args.figs,Path(__file__).stem))

        print(f'Processed {args.indices}, {len(indices) // 10:,d} images per class, {args.bins} bins')

    def create_entropies(self,images,selector,bins=20,m=32):
        '''
        Used to determine which pixels have the most information

        Parameters:
            images     Raw images from NIST
            selector   Indices of images that need to be included
            bins       Number of bins
            m          We will standardize images to be mxm
        '''
        n = len(selector)
        def create_1d_images():
            '''
            Convert images to be mxm, equalize, then convert to 1d
            '''
            m0,_ = images[0].shape
            product = np.zeros((n, m*m))
            for i in selector:
                right_sized_image = images[i] if m == m0 else resize(np.array(images[i]),(m,m))
                img = equalize_hist(right_sized_image)
                product[i] = np.reshape(img,-1)
            return product

        def create_entropies_from_1d_images(images1d):
            '''
            Calculate probability density for each pixel, then calculate entropy
            '''
            product = np.zeros((m*m))
            for i in range((m*m)):
                hist,edges = np.histogram(images1d[i],bins=bins,density=True)
                pdf = hist/np.sum(hist)
                product[i] = entropy(pdf)
            return product

        return create_entropies_from_1d_images(create_1d_images())

    def cull(self,img,n,mu,sigma,min0=0,clip=False):
        '''
        Cull data for display

        Parameters:
            img    An array of entropies, with one entry for each pixel
            n      Threshold for culling: number of standard deviations below mean
            mu     Mean entropy
            sigma  Standard deviation for entropy
            min0   Minimum entropy over all pixels
            clip   Set to true to set pixels to 1 if they survive culling
            ax     Axis for displaying data
        '''
        product = np.copy(img)
        product[product < mu + n*sigma] = min0
        if clip:
            product[product >= mu + n*sigma] = 1
        return product

    def show_image(self,entropies,ax=None,fig=None,cmap='Blues'):
        '''
        Show the distribution of entropies over images

        Parameters:
            entropies  An array of entropies, with one entry for each pixel
            ax         Axis for displaying data
            fig        Figure to be displayed
        '''
        fig.colorbar(
            ax.imshow(entropies,cmap=cmap),label='Entropy')
        ax.set_title('All pixels')

    def show_culled(self,entropies,n,mu,sigma,min0,ax=None,cmap='Blues'):
        '''
        Cull data and display

        Parameters:
            entropies  An array of entropies, with one entry for each pixel
            n          Threshold for culling: number of standard deviations below mean
            mu         Mean entropy
            sigma      Standard deviation for entropy
            min0       Minimum entropy over all pixels
            ax         Axis for displaying data
        '''
        ax.imshow(self.cull(entropies,n,mu,sigma,min0),cmap=cmap)
        ax.set_title(rf'Culled {abs(n)}$\sigma$ below $\mu$' if n != 0 else r'Culled all below $\mu$')

    def show_mask(self,mask,cmap='Blues',ax=None,size=28):
        ax.imshow(mask,cmap=cmap)
        ax.set_title(rf'Mask preserving {int(100*mask.sum()/(size**2))}\% of pixels')

    def show_histogram(self,entropies,mu,sigma,threshold=0.5,ax=None):
        '''
        Show histogram of entropies.

        Parameters:
            entropies  An array of entropies, with one entry for each pixel
            mu         Mean entropy
            sigma      Standard deviation for entropy
            ax         Axis for displaying data
        '''
        ax.hist(entropies.reshape(-1),bins='fd',density=True,color='xkcd:blue',label='Histogram')
        ax.set_xlabel('H')
        ax.set_ylabel('Frequency')
        ax.set_title('Entropy of pixels')
        ax.axvline(mu,c='xkcd:red',ls='-',label=r'$\mu$')
        ax.axvline(mu-threshold*sigma,c='xkcd:red',ls=':',label=r'$\mu' f'{-threshold}' r'\sigma$')
        ax.legend()

class EstablishSubsets(Command):
    '''
        Extract subsets of MNIST to facilitate replication
    '''
    def __init__(self):
        super().__init__('Establish Subsets','establish-subsets',needs_output_file=True)

    def _execute(self):
        indices = create_indices(self.ytrain, nimages=args.nimages, rng=self.rng)
        m, n = indices.shape
        file = Path(join(args.data, args.out)).with_suffix('.npy')
        np.save(file, indices)
        print(f'Saved {m} labels for each of {n} classes in {file.resolve()}')

class EstablishStyles(Command):
    '''
        Display representatives of all styles created by establish_styles.py
    '''
    def __init__(self):
        super().__init__('Establish Styles','establish-styles',needs_output_file=True)

    def _execute(self):
        n_examples, n_classes = self.indices.shape
        for i_class in args.classes:
            style_list = StyleList.build(self.x, self.indices,
                                         i_class=i_class,
                                         nimages=min(n_examples,args.nimages),
                                         threshold=args.threshold)
            print(f'Class {i_class} contains {len(style_list)} Styles')
            fig = figure(figsize=(8, 8))
            ax1 = fig.add_subplot(1, 1, 1)
            ax1.hist([len(style) for style in style_list.styles])
            ax1.set_title(f'Lengths of style for {len(style_list)} styles')
            fig.suptitle(f'Digit Class = {i_class}, threshold={self.args.threshold}')
            fig.savefig(join(args.figs, Path(__file__).stem + str(i_class)))
            file = Path(join(args.data, args.out+str(i_class))).with_suffix('.npy')
            style_list.save(file)

class DisplayStyles(Command):
    '''
        Display representatives of all styles created by establish_styles.py
    '''
    def __init__(self):
        super().__init__('Display Styles','display-styles')

    def _execute(self):
        indices = np.load(join(self.args.data, self.args.indices)).astype(int) # no need as already selected
        n_examples, n_classes = indices.shape

        for i_class in self.args.classes:
            fig = figure(figsize=(8, 8))
            x_class = self.x[self.indices[:,i_class],:]
            Allocation = np.load(join(self.args.data, self.args.styles+str(i_class)+'.npy')).astype(int)
            m1,n1 = Allocation.shape
            if self.args.nimages != None:
                n1 = min(n1,self.args.nimages)
            for j in range(m1):
                for k in range(n1):
                    ax = fig.add_subplot(m1, n1, j*n1 + k + 1)
                    img = x_class[Allocation[j,k]].reshape(args.size,args.size)
                    ax.imshow(img,cmap=cm.gray)



class Cluster(Command):
    def __init__(self):
        super().__init__('Cluster','explore-clusters')

    def _execute(self):
        fig = figure(figsize=(8, 8))
        indices = np.load(join(self.args.data,self.args.indices)).astype(int)
        n_examples,n_classes = indices.shape

        m,n = self.indices.shape  # images,classes
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
        fig.savefig(join(self.args.figs,Path(__file__).stem))

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

def parse_args(command_names):
    parser = ArgumentParser(__doc__)
    parser.add_argument('command',choices=command_names)
    parser.add_argument('out',nargs='?')
    parser.add_argument('--show', default=False, action='store_true', help='Controls whether plot will be displayed')
    parser.add_argument('--figs', default='./figs', help='Location for storing plot files')
    parser.add_argument('--data', default='./data', help='Location for storing data files')
    parser.add_argument('--indices', default='establish_subset.npy', help='Location where index files have been saved')
    parser.add_argument('--nimages', default=20, type=int, help='Maximum number of images for each class')
    parser.add_argument('--mask', default=None, help='Name of mask file (omit for no mask)')
    parser.add_argument('--size', default=28, type=int, help='Number of row/cols in each image: shape will be will be mxm')
    parser.add_argument('--classes', default=list(range(10)), type=int, nargs='+', help='List of digit classes')
    parser.add_argument('--bins', default=12, type=int, help='Number of bins for histograms')
    parser.add_argument('--threshold', default=0.1, type=float,  #FIXME
                        help='Include image in same style if mutual information exceeds threshold')

    group_display_styles = parser.add_argument_group('Options for display-styles')
    group_display_styles.add_argument('--styles', default=Path(__file__).stem, help='Location where styles have been stored')

    group_explore_clusters = parser.add_argument_group('Options for explore-clusters')
    group_explore_clusters.add_argument('--npairs', default=128, type=int, help='Number of pairs for each class')

    parser.add_argument('--cmap',default='Blues',help='Colour map') #FIXME

    parser.add_argument('--m', default=12, type=int, help='Number of images for each class')
    return parser.parse_args()


if __name__ == '__main__':
    rc('font', **{'family': 'serif',
                  'serif': ['Palatino'],
                  'size': 8})
    rc('text', usetex=True)
    start = time()
    Command.build([
        EstablishSubsets(),
        EstablishPixels(),
        EDA(),
        EstablishStyles(),
        DisplayStyles(),
        Cluster(),
        EDA_MI()
    ])
    args = parse_args(Command.get_command_names())
    command = Command.commands[args.command]
    command.set_args(args)
    command.execute()

    elapsed = time() - start
    minutes = int(elapsed / 60)
    seconds = elapsed - 60 * minutes
    print(f'Elapsed Time {minutes} m {seconds:.2f} s')

    if args.show:
        show()
