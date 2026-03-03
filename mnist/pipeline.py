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
    This program schedules commands to create files in the pipeline.
'''
from abc import ABC,abstractmethod
from argparse import ArgumentParser, RawDescriptionHelpFormatter
from textwrap import dedent
from pathlib import Path
from time import time, strftime,localtime
from shutil import copyfile
from matplotlib.pyplot import figure, show
from matplotlib import rc, cm
from matplotlib.ticker import MaxNLocator
import numpy as np
from skimage.exposure import equalize_hist
from skimage.transform import resize
from mnist import MnistDataloader, MnistException
from mask import Mask
from style import StyleList,StylesStoppedBuilding
from shared.utils import Logger,user_has_requested_stop,create_xkcd_colours,get_bins

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
            Command.commands[command.name] = command

    @staticmethod
    def get_names():
        '''
        Used to construct command line argument
        '''
        return [name for name in Command.commands.keys()]

    def __init__(self,description,name,
                 needs_output_file=False,
                 needs_index_file=True,
                 needs_style_file=False,
                 needs_likelihoods_file=False,
                 n=10):
        self.description = description
        self.name = name
        self.needs_output_file = needs_output_file
        self.needs_index_file = needs_index_file
        self.needs_style_file = needs_style_file
        self.needs_likelihoods_file = needs_likelihoods_file
        self.colours = create_xkcd_colours(n)

    def get_description(self):
        return self.description

    def set_args(self,args):
        self.args = args
        self.rng = np.random.default_rng(args.seed)

    def execute(self):
        '''
        Shared code for executing command:
        - Load mnist images and other data used by commands
        - apply mask
        '''
        self.log (f'{self.get_description()} {strftime("%a, %d %b %Y %H:%M:%S +0000", localtime())}')

        if self.needs_output_file and self.args.out == None:
            self.log ('Output file must be specified')
            exit(1)

        self.data_path = Path(self.args.data).resolve()
        self.figs_path = Path(self.args.figs).resolve()

        self.load_mnist_data()
        self.load_and_apply_mask()
        self.load_supplementary_files()

        self._execute()   # Perform actual command

    def load_mnist_data(self):
        '''
        Load training and test data
        '''
        dataloader = MnistDataloader.create(data=self.args.data,report = lambda x:self.log(x))
        (self.x_train, self.y_train), (self.x_test, self.y_test), = dataloader.load_data()
        self.x = MnistDataloader.columnize(self.x_train)

    def load_and_apply_mask(self):
        '''
        Read a mask from a file, if one is provided, and apply to training data
        If there is no mask file, set mask to all ones.
        '''
        self.mask, self.mask_text,self.bins = Mask.create(mask_file=self.args.mask,
                                                          data=self.args.data,
                                                          size=self.args.size,
                                                          report = lambda x:self.log(x))
        self.log('Bins: ' +str(self.bins),level=Logger.DEBUG)
        self.x = self.mask.apply(self.x)

    def load_supplementary_files(self):
        '''
        Load additional file needed by some commands
        '''
        if self.needs_index_file:
            file =  (self.data_path / self.args.indices).with_suffix('.npz')
            index_data = np.load(file)
            self.indices = index_data['indices']
            self.log (f'Loaded indices from {file}')

        if self.needs_style_file:
            file =  (self.data_path / self.args.styles).with_suffix('.npz')
            style_data = np.load(file,allow_pickle=True)
            self.Allocations = style_data['Allocations']
            self.log (f'Loaded Allocations from {file}')

        if self.needs_likelihoods_file:
            file =  (self.data_path / self.args.likelihoods).with_suffix('.npz')
            loaded_data = np.load(file,allow_pickle=True)
            self.class_styles = loaded_data['class_styles']
            self.A = loaded_data['A']
            self.log (f'Loaded Likelihoods from {file}')

    @abstractmethod
    def _execute(self):
        '''
        Execute command: must be implemented for each class
        '''
        ...

    def set_logger(self,logger):
        '''
        Attach a logger to Command
        '''
        self.logger = logger

    def log(self,message,level=Logger.INFO):
        '''
        Log messages
        '''
        self.logger.log(message,level=level)

    def digitize_images(self,imgs,equalize=False):
        '''
        Used by EstablishLikelihoods and RecognizeDigits to prepare images
        '''
        equalized_images = equalize_hist(imgs) if equalize else imgs
        return np.digitize(equalized_images,self.bins)

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
        Extract subsets of MNIST of a specified size, and save to index file
        '''
        indices = MnistDataloader.create_indices(self.y_train, nimages=self.args.nimages, rng=self.rng)
        file =  (self.data_path / self.args.out).with_suffix('.npz')
        np.savez(file, indices=indices)
        m,n = indices.shape
        self.log(f'Saved {m} labels for each of {n} classes in {file.resolve()}')

class EstablishMask(Command):
    '''
    Determine which pixels are most relevant to classifying images
    '''
    def __init__(self):
        super().__init__('Establish Mask','establish-mask',needs_output_file=True)

    def _execute(self):
        '''
        Determine which pixels are most relevant to classifying images
        '''
        indices = self.indices.reshape(-1)
        mask = Mask.build(np.array(self.x_train),indices,bins=self.args.bins,m=self.args.size,fraction=self.args.fraction)
        bins = self._plot(mask,indices)
        file =  (self.data_path / self.args.out).with_suffix('.npz')
        mask.save(file, bins=bins)

        self.log(f'Processed {self.args.indices}, {len(indices) // 10:,d} images per class, {self.args.bins} bins, saved mask in {file}')

    def _plot(self,mask,indices):
        fig = figure(figsize=(12, 12))

        ax1 = fig.add_subplot(2,3,1)
        mappable = ax1.imshow(mask.img,cmap=self.args.cmap)
        fig.colorbar(mappable,label='Entropy')
        ax1.set_title('All pixels')

        ax2 = fig.add_subplot(2,3,2)
        ax2.imshow(mask.img*mask.pixels,cmap=self.args.cmap)
        ax2.set_title('Culled pixels')

        bins = self.show_pixels(mask.pixels,ax=fig.add_subplot(2,3,3),bins=self.args.bins)

        ax4 = fig.add_subplot(2,3,4)
        ax4.imshow(mask.pixels,cmap=self.args.cmap)
        ax4.set_title(rf'Mask preserving {int(100*mask.pixels.sum()/(self.args.size*self.args.size))}\% of pixels')

        ax5 = fig.add_subplot(2,3,5)
        ax5.hist(mask.entropies,bins=args.bins,density=True,color='xkcd:blue',label='Histogram')
        ax5.set_xlabel('H')
        ax5.set_ylabel('Frequency')
        ax5.set_title('Entropy of pixels')
        if self.args.fraction == 0:
            ax5.axvline(mask.mu,c='xkcd:red',ls='-',label=r'$\mu=$' f'{mask.mu:.3f} (threshold)')
        else:
            ax5.axvline(mask.mu,c='xkcd:red',ls='-',label=r'$\mu=$' f'{mask.mu:.3f}')
            ax5.axvline(mask.threshold,c='xkcd:red',ls=':',label=f'Threshold={mask.threshold:.3f}')

        ax5.legend()

        fig.suptitle(r'Processed \emph{' + f'{self.args.indices}'
                     ',} '  f'{len(indices) // 10:,d} images per class, {self.args.bins} bins')
        fig.savefig((self.figs_path / self.args.out).with_suffix('.png'))

        return bins

    def show_pixels(self,mask,ax=None,bins=10):
        '''
        Show histogram of pixel intensity

        Parameters:
            mask    The mask, identifying which pixels contain the most information
            ax      Axis for plotting
            bins    Number of bins to histogram

        Returns:
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
        _,bin_edges,_ = ax.hist(pixels,bins=bins,facecolor='xkcd:red',alpha=1.0,label='Included in mask',hatch='/', edgecolor='k')
        ax.legend()
        ax.set_xlabel('Intensity')
        ax.set_ylabel('Frequency')
        ax.set_title('Intensity of pixels')
        return bin_edges

class EstablishStyles(Command):
    '''
    Display representatives of all styles created by establish_styles.py
    '''
    def __init__(self):
        super().__init__('Establish Styles','establish-styles',needs_output_file=True)


    def _execute(self):
        '''
        Allocate exemplars to styles
        '''
        n_examples, n_classes = self.indices.shape
        N = np.zeros((self.args.nimages,len(self.args.classes)))
        L = 0
        max_steps = -1
        Allocations = self.create_allocations()
        fig = figure(figsize=(12, 8))

        for j,i_class in enumerate(self.args.classes):
            if user_has_requested_stop(): break
            try:
                style_list,steps = StyleList.build(self.x, self.indices,
                                                   i_class=i_class,
                                                   nimages=min(n_examples,self.args.nimages),
                                                   threshold=self.args.threshold)

                Allocations[i_class] = style_list.create_allocations()
                self.plot_lengths(style_list,i_class, ax = fig.add_subplot(3, 4, 1+j)  )
                for i in steps:
                    N[i+1:,j] += 1
                max_steps = max(max_steps,steps[-1])

                file =  (self.data_path / self.args.out).with_suffix('.npz')
                np.savez(file,Allocations=Allocations)
                self.log(f'Class {i_class} contains {len(style_list)} Styles, saved styles in {file}')
            except StylesStoppedBuilding:
                self.log(f'Stopped while processing {i_class}')
                break
        try:
            self.plot_styles_versus_exemplars(max_steps,N, fig=fig)
            fig.tight_layout(pad=2,h_pad=2,w_pad=2)
            fig.savefig((self.figs_path / self.args.out).with_suffix('.png'))
        except ValueError:
            return

    def create_allocations(self):
        '''
        Create array of Allocations, either by reading from a file or from scratch.

        Returns:
            A vector with one element per digit class
        '''
        if self.args.restart:
            file =  (self.data_path / self.args.styles).with_suffix('.npz')
            style_data = np.load(file,allow_pickle=True)
            self.log (f'Loaded Allocations from {file}')
            self.verify_ok_to_write(file)
            return style_data['Allocations']
        else:
            return np.empty((10),dtype=np.ndarray)

    def verify_ok_to_write(self,file):
        '''
        Verify that we won't lose data when we write to output file,
        by creating backup file if necessary

        Parameters:
            file     Full path name for styles
        '''
        if Path(self.args.out).stem == file.stem:
            copyfile( file,file.with_suffix('.bak'))

    def plot_lengths(self,style_list,i_class,ax=None):
        '''
        Plot histogram of lengths of styles within list
        '''
        ax.hist([len(style) for style in style_list.styles],color=self.colours[i_class])
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_title(f'Digit Class {i_class}, Lengths for {len(style_list)} styles')
        ax.set_xlabel('Length')

    def plot_styles_versus_exemplars(self,max_steps,N,fig=None):
        '''
        Plot the length of each style as a function of number of exemplars
        '''
        ax1 = fig.add_subplot(3, 4, 11)
        for j,i_class in enumerate(self.args.classes):
            ax1.plot(list(range(max_steps)),N[0:max_steps,j],label={i_class},c=self.colours[i_class])
        ax1.set_xlabel('Number of exemplars')
        ax1.set_ylabel('Number of styles')
        ax1.set_title('Style Learning')
        handles, labels = ax1.get_legend_handles_labels()

        ax2 = fig.add_subplot(3, 4, 12)
        ax2.legend(handles, labels, loc='center left', frameon=False,title='Classes')
        ax2.axis('off')


class EstablishLikelihoods(Command):
    '''
    Calculate the Likelihood matrices and save text
    '''
    def __init__(self):
        super().__init__('Calculate the Likelihood matrices','establish-likelihoods',
                         needs_output_file=True,
                         needs_style_file=True)

    def _execute(self):
        '''
        For each pixel, determine the probability of belonging to each digit and style
        '''
        file =  (self.data_path / self.args.out).with_suffix('.npz')
        class_styles,starting_positions = self.build_class_style_mapping()
        np.savez(file,
                 A=self.create_likelihoods(starting_positions,class_styles),
                 class_styles=class_styles)
        self.log (f'Saved Likelihoods and class/styles from {(self.data_path / self.args.styles).with_suffix('.npz')} in {file}')

    def build_class_style_mapping(self):
        '''
        Construct mapping between class/style and position in the Likelihood matrix,A

        Returns:
            class_styles        An array, each row representing a digit class and a style.
                                [0,0],...[0,n-1],[1,n],..., where n is the number of styles in class zero.
                                So the row index of [i,j] is the position of class i style j in the
                                Likelihood matrix
            starting_positions  The positions in the Likelihood matrix corresponding to the first
                                style in each digit class
        '''
        class_styles = []
        starting_positions = np.zeros(len(self.args.classes),dtype=int)

        for i_class in self.args.classes:
            n_styles,_ = self.Allocations[i_class].shape
            position_start = len(class_styles)
            starting_positions[i_class] = position_start
            run_for_this_class = [[i_class,position_start + i] for i in range(n_styles)]
            class_styles += run_for_this_class

        return np.array(class_styles),starting_positions

    def create_likelihoods(self,starting_positions,class_styles,n_pixels = 28*28 ): #FIXME mask?
        '''
        Add up pixels for each combination of class,style and normalize.
        '''
        n_class_styles,_ = class_styles.shape
        A = self.args.pseudocount*np.ones((n_class_styles,n_pixels,len(self.bins)+1))
        for i_class in self.args.classes:
            digitized_images = self.digitize_images(self.x[self.indices[:,i_class],:])
            _,n_pixels = digitized_images.shape

            # The Allocations vector has one element for each digit class;
            # Allocations[i_class] is a matrix with one row for each style , and sufficient columns to
            # represent all images in the most numerous style in the class. Each element contains the
            # index of one images from the style, or -1 if there are insufficient images to fill the row.
            n_styles,n_images_for_style = self.Allocations[i_class].shape
            for i_style in range(n_styles):                  # Accumulate counts for all styles
                i_class_style = starting_positions[i_class] + i_style
                for image_seq in range(n_images_for_style):  # Accumulate counts for all images in specific style
                    image_index = self.Allocations[i_class][i_style,image_seq]
                    if image_index > -1:      # skip unassigned positions
                        this_image = digitized_images[image_index]
                        for j in range(n_pixels):
                            if self.mask[j] and this_image[j] > 1: # No vote if pixel is zero
                                A[i_class_style,j,this_image[j]] += 1

        Evidence = A.sum(axis=0)

        return A / Evidence

class RecognizeDigits(Command):
    '''
    Use A matrices to recognize class
    '''
    def __init__(self):
        super().__init__('Use likelihood matrices to recognize class','recognize-digits',
                         needs_likelihoods_file=True,needs_output_file=True)

    def _execute(self):
        '''
        For each pixel, determine the probability of belonging to each digit and style
        '''
        self.logA = np.log(self.A)

        N,accuracy,mismatches = self.get_accuracy(self.x_test,self.y_test)
        self.log (f'{N} images, accuracy={accuracy}')
        self.plot_mismatches(min(N,args.max_images),accuracy,mismatches)

    def plot_mismatches(self,N,accuracy,mismatches):
        '''
        Plot mismatched images, using mask as background (so we can be sure we aren't losing important information)

        Parameters:
            N           Number of images that we used for prediction
            accuracy    Overall accuracy
            mismatches  Array of indices of mismatched predictions
        '''
        fig = figure(figsize=(8,8))
        m,n = get_subplot_shape(len(mismatches))
        for k,(img,y,prediction) in enumerate(mismatches):
            ax = fig.add_subplot(m,n,k+1)
            ax.imshow(self.mask.pixels,cmap='Reds')
            ax.imshow(img,cmap=self.args.cmap,alpha=0.5)
            ax.axis('off')
            ax.set_title(f'{prediction} ({y})')

        fig.suptitle(f'{N} images, accuracy={int(100*accuracy)}\\%')
        fig.tight_layout(pad=2,h_pad=2,w_pad=2)
        fig.savefig((self.figs_path / self.args.out).with_suffix('.png'))

    def predict(self,img,nclasses=10):
        '''
        Compute the probability of each digit as a cause for image.
        We will accumulate the posterior probilities for each
        style within each class,

        Parameters:
            img
            nclasses
        '''
        digitized_image = self.digitize_images(img.reshape(-1))
        K,J,_ = self.A.shape
        log_posterior_for_styles = np.zeros((K))
        for k in range(K):
            for j in range(J):
                if self.mask[j]:
                    log_posterior_for_styles[k] += self.logA[k,j,digitized_image[j]]

        i = np.argmax(log_posterior_for_styles)
        return self.class_styles[i,0],i,log_posterior_for_styles

    def get_accuracy(self,x,y):
        '''
        Compute accuracy of predictions: predict the class of each image,
        and compare with label.

        Parameters:
            x        Array of images
            y        Array of expected labels
        '''
        matches = 0
        mismatches = []
        N = len(y)
        if self.args.N != None:
            N = min(N,self.args.N)
        Selection = self.rng.choice(len(y),N,replace=False) if N < len(y) else range(N)
        for i in Selection:
            prediction,index,log_posterior_for_styles = self.predict(np.array(x[i]))
            if y[i] == prediction:
                matches += 1
            else:
                mismatches.append((x[i],y[i],prediction))
                self.log(f'y={y[i]}, prediction={prediction}, index={index}',level=Logger.DEBUG)
                self.log(str(log_posterior_for_styles),level=Logger.DEBUG)

        return N,matches/N,mismatches


def get_subplot_shape(N):
    '''
    Determine the number of rows and columns needed for a specified number of subplots

    Parameters:
        N       Number of subplots

    Returns:
       m     Number of rows
       n     Numbers of columns

    Post Condition:
        m*n >= N, n >= m, and n - m is as small as possible
    '''
    m = int(np.sqrt(N))
    n = N // m
    while m*n < N:
        n += 1
    return m,n

def parse_args(names):
    parser = ArgumentParser(__doc__)
    parser.add_argument('command',choices=names,help='The command to be executed')
    parser.add_argument('-o','--out',nargs='?')
    parser.add_argument('--show', default=False, action='store_true', help='Controls whether plot will be displayed')
    parser.add_argument('--figs', default='./figs', help='Location for storing plot files')
    parser.add_argument('--data', default='./data', help='Location for storing data files')
    parser.add_argument('--indices', default=None, help='Location where index files have been saved')
    parser.add_argument('--nimages', default=2000, type=int, help='Maximum number of images for each class')
    parser.add_argument('--mask', default=None, help='Name of mask file (omit for no mask)')
    parser.add_argument('--size', default=28, type=int, help='Number of row/cols in each image: shape will be will be mxm')
    parser.add_argument('--classes', default=list(range(10)), type=int, nargs='+', help='List of digit classes')
    parser.add_argument('--seed', default=None, type=int, help='For initializing random number generator')
    parser.add_argument('--cmap',default='Blues',help='Colour map')
    parser.add_argument('--logs', default='./logs', help='Location for storing log files')
    parser.add_argument('--styles', default=None, help='Location where styles have been stored')

    group_establish_mask = parser.add_argument_group('Options for Establish mask')
    group_establish_mask.add_argument('--fraction', default=0.0, type=float,
                        help='Include pixel if entropy exceeds mean - fraction*sd')
    parser.add_argument('--bins', default='doane', type=get_bins, help='Number of bins for histograms')

    group_establish_styles = parser.add_argument_group('Options for establish-styles')
    group_establish_styles.add_argument('--threshold', default=0.1, type=float,
                          help='Include image in same style if mutual information exceeds threshold')
    group_establish_styles.add_argument('--restart',default=False,action='store_true',help='Continue processing styles')

    group_calculate_A = parser.add_argument_group('Options for calculate-likelihoods')
    group_calculate_A.add_argument('--pseudocount', default=0.05, type=float,help='Used to initialize counts')

    group_recognize = parser.add_argument_group('Options for recognize')
    group_recognize.add_argument('--likelihoods', default='A.npz', help='Location where A matrices files have been saved')
    group_recognize.add_argument('--N', default=None,type=int, help='Number of images for calculating accuracy')
    group_recognize.add_argument('--max_images', default=100,type=int, help='Maximum number of images')

    return parser.parse_args()

if __name__ == '__main__':
    rc('font', **{'family': 'serif',
                  'serif': ['Palatino'],
                  'size': 8})
    rc('text', usetex=True)
    start = time()
    Command.build([
        EstablishSubsets(),
        EstablishMask(),
        EstablishStyles(),
        EstablishLikelihoods(),
        RecognizeDigits()
    ])
    args = parse_args(Command.get_names())
    with Logger(Path(__file__).stem,path=args.logs) as logger:
        command = Command.commands[args.command]
        command.set_args(args)
        command.set_logger(logger)
        code = 0
        try:
            command.execute()
        except FileNotFoundError as e:
            command.log(f'Error: {e.filename} not found.',level=Logger.ERROR)
            code = 1
        except MnistException as e:
            command.log('MnistException {e}',level=Logger.ERROR)
            code = 1
        finally:
            elapsed = time() - start
            minutes = int(elapsed / 60)
            seconds = elapsed - 60 * minutes
            logger.log(f'Elapsed Time {minutes} m {seconds:.2f} s')
            if code > 0: exit(code)

            if args.show:
                show()
