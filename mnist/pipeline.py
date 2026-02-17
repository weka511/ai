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
    This program schedules commands to create files in pipeline.
'''
from abc import ABC,abstractmethod
from argparse import ArgumentParser, RawDescriptionHelpFormatter
from textwrap import dedent
from os.path import join
from pathlib import Path
from time import time, strftime,localtime
from matplotlib.pyplot import figure, show
from matplotlib import rc, cm
from matplotlib.ticker import MaxNLocator
import numpy as np
from skimage.exposure import equalize_hist
from sklearn.feature_selection import mutual_info_classif
from skimage.transform import resize
from pymdp.maths import softmax
from mnist import MnistDataloader, create_mask, columnize,create_indices,create_entropies
from style import StyleList

class Command(ABC):
    '''
    Parent class for procesing requests
    '''
    commands = {}

    @staticmethod
    def build(command_list):
        '''
        Load a list of commands that are available for execution
        '''
        for command in command_list:
            Command.commands[command.command_name] = command

    @staticmethod
    def get_command_names():
        '''
        Used to construct command line argument
        '''
        return [name for name in Command.commands.keys()]

    @staticmethod
    def get_command_help():
        '''
        Used to construct command line argument
        '''
        return ''.join([f'{key}\t{value.name}\n' for key,value in Command.commands.items()])

    def __init__(self,name,command_name,needs_output_file=False,needs_index_file=True):
        self.name = name
        self.command_name = command_name
        self.needs_output_file = needs_output_file
        self.needs_index_file = needs_index_file

    def get_name(self):
        return self.name

    def set_args(self,args):
        self.args = args
        self.rng = np.random.default_rng(args.seed)

    def execute(self):
        '''
        Shared code for executing command:
        - Load mnist images and other data used by commands
        - apply mask
        '''
        print (self.get_name(),strftime("%a, %d %b %Y %H:%M:%S +0000", localtime()))
        if self.needs_output_file and args.out == None:
            print ('Output file must be specified')
            exit(1)
        mnist_dataloader = MnistDataloader.create(data=self.args.data)
        (self.x_train, self.y_train), (self.x_test, self.y_test), = mnist_dataloader.load_data()
        x = columnize(self.x_train)

        self.mask, self.mask_text,self.n,self.bins = create_mask(mask_file=self.args.mask, data=self.args.data, size=self.args.size)
        self.mask = self.mask.reshape(-1)
        self.x = np.multiply(x, self.mask)

        if self.needs_index_file:
            file = Path(join(self.args.data, self.args.indices)).with_suffix('.npz')
            index_data = np.load(file)
            self.indices = index_data['indices']
            print (f'Loaded indices from {file}')

        self._execute()   # Perform actual command

    @abstractmethod
    def _execute(self):
        '''
        Execute command: must be implemented for each class
        '''
        ...

    def load_allocations(self):
        file = Path(join(self.args.data, self.args.styles)).with_suffix('.npz')
        style_data = np.load(file,allow_pickle=True)
        Allocations = style_data['Allocations']
        print (f'Load Allocations from {file}')
        return Allocations

class EstablishSubsets(Command):
    '''
    Extract subsets of MNIST to facilitate replication
    '''

    def __init__(self):
        super().__init__('Establish Subsets','establish-subsets',
                         needs_output_file=True,
                         needs_index_file=False)

    def _execute(self):
        '''
        Extract subsets of MNIST and save to index file
        '''
        indices = create_indices(self.y_train, nimages=args.nimages, rng=self.rng)
        file = Path(join(args.data, args.out)).with_suffix('.npz')
        np.savez(file, indices=indices)
        m,n = indices.shape
        print(f'Saved {m} labels for each of {n} classes in {file.resolve()}')

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

        MI_within_classes = np.zeros((n_classes,args.nimages))
        for i in range(n_classes):
            companions = self.create_companions(i,n_comparison=args.nimages)
            MI_within_classes[i] = mutual_info_classif(companions.T,Exemplars[i,:])

        fig = figure(figsize=(20, 8))
        ax1 = fig.add_subplot(1,2,1)
        fig.colorbar(ax1.imshow(MI_between_classes, cmap=args.cmap, interpolation='nearest'),
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


class EstablishPixels(Command):
    '''
    Determine which pixels are most relevant to classifying images
    '''
    def __init__(self):
        super().__init__('Establish Pixels','establish-pixels',needs_output_file=True)

    def _execute(self):
        '''
        Determine which pixels are most relevant to classifying images
        '''
        x_train = np.array(self.x_train)
        indices = self.indices.reshape(-1)
        entropies = create_entropies(x_train[indices],
                                     list(range(len(indices))),
                                     bins=args.bins,
                                     m=args.size)
        mu = np.mean(entropies)
        sigma = np.std(entropies)
        min0 = np.min(entropies)
        img = np.reshape(entropies,(args.size,args.size))
        mask = self.cull(entropies,-args.fraction,mu,sigma,clip=True).reshape(args.size,args.size)

        fig = figure(figsize=(12, 12))
        self.show_image(img,ax=fig.add_subplot(2,3,1),fig=fig,cmap=args.cmap)
        self.show_culled(img,-args.fraction,mu,sigma,min0,ax = fig.add_subplot(2,3,2),cmap=args.cmap)
        self.show_mask(mask,cmap=args.cmap,ax = fig.add_subplot(2,3,4),size=args.size)
        self.show_histogram(img,mu,sigma,threshold=args.fraction,ax=fig.add_subplot(2,3,5))
        n,bins = self.show_pixels(mask,ax=fig.add_subplot(2,3,3))
        fig.suptitle(r'Processed \emph{' + f'{args.indices}'
                     ',} '  f'{len(indices) // 10:,d} images per class, {args.bins} bins')
        fig.savefig(join(args.figs,args.out))

        file = Path(join(args.data, args.out)).with_suffix('.npz')
        np.savez(file, mask=mask,n=n,bins=bins)

        print(f'Processed {args.indices}, {len(indices) // 10:,d} images per class, {args.bins} bins, saved mask in {file}')

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
            threshold  Thrshold for culling: we represent this by a verticl line
        '''
        ax.hist(entropies.reshape(-1),bins='fd',density=True,color='xkcd:blue',label='Histogram')
        ax.set_xlabel('H')
        ax.set_ylabel('Frequency')
        ax.set_title('Entropy of pixels')
        ax.axvline(mu,c='xkcd:red',ls='-',label=r'$\mu$')
        ax.axvline(mu-threshold*sigma,c='xkcd:red',ls=':',label=r'$\mu' f'{-threshold}' r'\sigma$')
        ax.legend()

    def show_pixels(self,mask,ax=None,bins=10):
        '''
        Show histogram of pixel intensity

        Parameters:
            mask    The mask, identifying which pixels contain the most information
            ax      Axis for plotting
            bins    Number of bins to histogram

        Returns:
            n          Counts in each bin
            bin_edges  The edges of the bins that were actually used
        '''
        def generate_images():
            '''
            Used to iterate over all images that have been sampled via Establish Subsets
            '''
            m,n = self.indices.shape
            for i in range(m):
                for j in range(n):
                    yield self.indices[i,j]

        def collect_pixels():
            '''
            Use mask to segregate pixels into in-group and out-group

            Returns:
                Values of pixels that will be included
                Values of pixels that will be masked out
            '''
            pixels = []
            masked_out = []
            for index in generate_images():
                img = equalize_hist(np.array(self.x_train[index]))
                m,n = img.shape
                for i in range(m):
                    for j in range(n):
                        if mask[i,j] == 1:
                            pixels.append(img[i,j])
                        else:
                            masked_out.append(img[i,j])

            return pixels,masked_out

        pixels,masked_out = collect_pixels()
        ax.hist(pixels + masked_out,bins=bins,color='xkcd:blue',alpha=1.0,label='All')
        n,bin_edges,_ = ax.hist(pixels,bins=bins,facecolor='xkcd:red',alpha=1.0,label='Included in mask',hatch='/', edgecolor='k')
        ax.legend()
        ax.set_xlabel('Intensity')
        ax.set_ylabel('Frequency')
        ax.set_title('Intensity of pixels')
        return n,bin_edges

class EstablishStyles(Command):
    '''
    Display representatives of all styles created by establish_styles.py
    '''
    def __init__(self):
        super().__init__('Establish Styles','establish-styles',needs_output_file=True)
        self.colours = ['lightgreen', 'orange','teal',
                        'lightblue', 'red', 'brown',
                        'pink',	'blue', 'green', 'purple'
        ]

    def _execute(self):
        '''
        Allocate exemplars to styles
        '''
        n_examples, n_classes = self.indices.shape
        N = np.zeros((args.nimages,len(args.classes)))
        L = 0
        max_steps = -1
        Allocations = np.empty((10),dtype=np.ndarray)
        fig = figure(figsize=(12, 8))
        for j,i_class in enumerate(self.args.classes):
            style_list,steps = StyleList.build(self.x, self.indices,
                                               i_class=i_class,
                                               nimages=min(n_examples,self.args.nimages),
                                               threshold=self.args.threshold)

            Allocations[j] = style_list.create_allocations()
            print(f'Class {i_class} contains {len(style_list)} Styles.')
            self.plot_lengths(style_list,i_class, ax = fig.add_subplot(3, 4, 1+j)  )
            for i in steps:
                N[i+1:,j] += 1
            max_steps = max(max_steps,steps[-1])

        file = Path(join(self.args.data, self.args.out)).with_suffix('.npz')
        np.savez(file,Allocations=Allocations)
        print (f'Saved styles in {file}')
        self.plot_styles_versus_exemplars(max_steps,N, fig=fig)
        fig.tight_layout(pad=2,h_pad=2,w_pad=2)
        fig.savefig(Path(join(self.args.figs, self.args.out)).with_suffix('.png'))

    def plot_lengths(self,style_list,i_class,ax=None):
        '''
        Plot histogram of lengths of styles within list
        '''
        ax.hist([len(style) for style in style_list.styles],color='xkcd:'+self.colours[i_class])
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_title(f'Digit Class {i_class}, Lengths for {len(style_list)} styles')
        ax.set_xlabel('Length')

    def plot_styles_versus_exemplars(self,max_steps,N,fig=None):
        '''
        Plot the length of each style as a function of number of exemplars
        '''
        ax1 = fig.add_subplot(3, 4, 11)
        for j,i_class in enumerate(self.args.classes):
            ax1.plot(list(range(max_steps)),N[0:max_steps,j],label={i_class},c='xkcd:'+self.colours[j])
        ax1.set_xlabel('Number of exemplars')
        ax1.set_ylabel('Number of styles')
        ax1.set_title('Style Learning')
        handles, labels = ax1.get_legend_handles_labels()

        ax2 = fig.add_subplot(3, 4, 12)
        ax2.legend(handles, labels, loc='center left', frameon=False,title='Classes')
        ax2.axis('off')

class DisplayStyles(Command):
    '''
    Display representatives of all styles created by EstablishStyles
    '''
    def __init__(self):
        super().__init__('Display Styles','display-styles')


    def _execute(self):
        '''
        Display representatives of all styles created by establish-styles
        '''
        Allocations = self.load_allocations()

        for i_class in self.args.classes:
            fig = figure(figsize=(8, 8))
            x_class = self.x[self.indices[:,i_class],:]
            n_styles,n_images = Allocations[i_class].shape
            if self.args.nimages != None:
                n_images = min(n_images,self.args.nimages)
            n_styles=min(n_styles,7)  #FIXNE
            for j in range(n_styles):
                for k in range(n_images):
                    ax = fig.add_subplot(n_styles, n_images, j*n_images + k + 1)
                    img = x_class[Allocations[i_class][j,k]].reshape(args.size,args.size)
                    ax.imshow(img,cmap=args.cmap)
                    ax.axis('off')
            fig.tight_layout(pad=2,h_pad=2,w_pad=2)
            fig.savefig(Path(join(self.args.figs, self.args.styles+str(i_class))).with_suffix('.png'))

class CalculateLikelihoods(Command):
    '''
    Calculate the A matrices
    '''
    def __init__(self):
        super().__init__('Calculate the Likelihood matrices','calculate-likelihood',needs_output_file=True)

    def _execute(self):
        '''
        For each pixel, determine the probability of belonging to each digit and style
        '''
        index_style_start,class_styles = self.create_class_styles()
        A = self.create_A(index_style_start,class_styles)
        file = Path(join(args.data, args.out)).with_suffix('.npz')
        np.savez(file,A=A,class_styles=class_styles)
        print (f'Saved A and class_styles from {join(self.args.data, self.args.styles)} in {file}')

    def create_class_styles(self):
        '''
        Create mapping between class/style and position in A matrix
        '''
        product = []
        index_style_start = np.zeros(len(self.args.classes),dtype=int)  # FIXME - duplicate code
        style_data = np.load(Path(join(self.args.data, self.args.styles)).with_suffix('.npz'),allow_pickle=True)
        self.Allocations = style_data['Allocations']
        for i_class in self.args.classes:
            x_class = self.x[self.indices[:,i_class],:]
            _,n_pixels = x_class.shape
            n_styles,n_images = self.Allocations[i_class].shape
            index_style_start[i_class] = len(product)
            for i in range(n_styles):
                product.append([i_class,int(i + index_style_start[i_class])]) # avoid messy np.int64

        return index_style_start,np.array(product)

    def create_A(self,index_style_start,class_styles,n_pixels = 784 ): #FIXME
        '''
        Add up pixels for each combination of class,style and normalize
        '''
        n_class_styles,_ = class_styles.shape
        A = args.pseudocount*np.ones((n_class_styles,n_pixels,len(self.bins)+1))  # FIXME
        for i_class in self.args.classes:
            eq = equalize_hist(self.x[self.indices[:,i_class],:])
            x_class = np.digitize(eq,self.bins)
            _,n_pixels = x_class.shape
            n_styles,n_images = self.Allocations[i_class].shape
            for i_style in range(n_styles):
                for image_seq in range(n_images):
                    image_index = self.Allocations[i_class][i_style,image_seq]
                    if image_index < 0: break
                    img = x_class[image_index]
                    i = index_style_start[i_class] + i_style
                    for j in range(n_pixels):
                        A[i,j,img[j]] += 1

        return A/ A.sum(axis=0)

class Recognize(Command):
    '''
    Use A matrices to recognize class
    '''
    def __init__(self):
        super().__init__('Use A matrices to recognize class','recognize')

    def _execute(self):
        '''
        For each pixel, determine the probability of belonging to each digit and style
        '''
        loaded_data = np.load(join(self.args.data,self.args.A)) #FIXME #51
        self.class_styles = loaded_data['class_styles']
        self.A = loaded_data['A']
        print ( self.get_accuracy(self.x_train,self.y_train))
        print ( self.get_accuracy(self.x_test,self.y_test))

    def predict(self,img,nclasses=10):
        '''
        Compute the probability of each digit as a cause for image.
        We will accumulate the posterior probilities for each
        style within each class,
        '''
        posterior_for_styles = self.A @ equalize_hist(img.reshape(-1))  # Not normalized
        predictions = np.zeros((nclasses))
        m,_ = self.class_styles.shape
        for i in range(m):
            iclass = self.class_styles[i,0]
            istyle = self.class_styles[i,1]   # Not used...see Issue #49
            predictions[iclass] += posterior_for_styles[i]

        return softmax(predictions)

    def get_accuracy(self,x,y):
        '''
        Compute accuracy of predictions: predict the class of each image,
        and compare with label.
        '''
        matches = 0
        N = len(y)
        for i in range(N):
            predictions = self.predict(np.array(x[i]))
            if y[i] == np.argmax(predictions):  matches += 1

        return N,matches/N

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
        fig.savefig(join(self.args.figs,args.out))

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

def parse_args(command_names,text):
    parser = ArgumentParser(__doc__,formatter_class=RawDescriptionHelpFormatter, epilog=dedent(text))
    parser.add_argument('command',choices=command_names,help='The command to be executed')
    parser.add_argument('-o','--out',nargs='?')
    parser.add_argument('--show', default=False, action='store_true', help='Controls whether plot will be displayed')
    parser.add_argument('--figs', default='./figs', help='Location for storing plot files')
    parser.add_argument('--data', default='./data', help='Location for storing data files')
    parser.add_argument('--indices', default='baseline.npz', help='Location where index files have been saved')
    parser.add_argument('--nimages', default=1000, type=int, help='Maximum number of images for each class')
    parser.add_argument('--mask', default=None, help='Name of mask file (omit for no mask)')
    parser.add_argument('--size', default=28, type=int, help='Number of row/cols in each image: shape will be will be mxm')
    parser.add_argument('--classes', default=list(range(10)), type=int, nargs='+', help='List of digit classes')
    parser.add_argument('--bins', default=12, type=int, help='Number of bins for histograms')
    parser.add_argument('--seed', default=None, type=int, help='For initializing random number generator')
    parser.add_argument('--cmap',default='Blues',help='Colour map')

    group_eda = parser.add_argument_group('Options for eda')
    group_eda.add_argument('--images_per_digit',default=8,type=int,help='Number of images in each digit class')

    group_establish_pixels = parser.add_argument_group('Options for establish-pixels')
    group_establish_pixels.add_argument('--fraction', default=0.5, type=float,
                        help='Include pixel if entropy exceeds mean - fraction*sd')

    group_establish_styles = parser.add_argument_group('Options for establish-styles')
    group_establish_styles.add_argument('--threshold', default=0.1, type=float,
                          help='Include image in same style if mutual information exceeds threshold')

    group_display_styles = parser.add_argument_group('Options for display-styles')
    group_display_styles.add_argument('--styles', default=Path(__file__).stem, help='Location where styles have been stored')

    group_explore_clusters = parser.add_argument_group('Options for explore-clusters')
    group_explore_clusters.add_argument('--npairs', default=128, type=int, help='Number of pairs for each class')

    group_calculate_A = parser.add_argument_group('Options for calculate-A')
    group_calculate_A.add_argument('--pseudocount', default=0.5, type=float,help='Used to initialize counts')

    group_recognize = parser.add_argument_group('Options for recognize')
    group_recognize.add_argument('--A', default='A.npz', help='Location where A matrices files have been saved')

    return parser.parse_args()


if __name__ == '__main__':
    rc('font', **{'family': 'serif',
                  'serif': ['Palatino'],
                  'size': 8})
    rc('text', usetex=True)
    start = time()
    Command.build([
        EstablishSubsets(),
        EDA(),
        EstablishPixels(),
        EDA_MI(),
        Cluster(),
        EstablishStyles(),
        DisplayStyles(),
        CalculateLikelihoods(),
        Recognize()
    ])
    args = parse_args(Command.get_command_names(),Command.get_command_help())
    command = Command.commands[args.command]
    command.set_args(args)
    try:
        command.execute()
    except FileNotFoundError as e:
        print(f'Error: {e.filename} not found.')
        exit (1)

    elapsed = time() - start
    minutes = int(elapsed / 60)
    seconds = elapsed - 60 * minutes
    print(f'Elapsed Time {minutes} m {seconds:.2f} s')

    if args.show:
        show()
