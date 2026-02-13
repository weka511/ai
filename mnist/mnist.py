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
    This module contains functions to read and sample MNIST data.
'''

from argparse import ArgumentParser
from os.path import join
from pathlib import Path
import struct
from array import array
from time import time
import numpy as np
from scipy.stats import entropy
from matplotlib.pyplot import figure, show
from matplotlib import rc, cm
from skimage.exposure import equalize_hist
from skimage.transform import resize
from sklearn.feature_selection import mutual_info_classif

class MnistDataloader(object):
    '''
    Read MNIST data

    snarfed from https://www.kaggle.com/code/hojjatk/read-mnist-dataset

    Data Members:
        training_images_filepath
        training_labels_filepath
        test_images_filepath
        test_labels_filepath
    '''
    @staticmethod
    def create(data = './data'):
        '''
        Create a MnistDataloader, setting up pathnames for all datasets
        '''
        return MnistDataloader(
            training_images_filepath = join(data, 'train-images-idx3-ubyte/train-images-idx3-ubyte'),
            training_labels_filepath = join(data, 'train-labels-idx1-ubyte/train-labels-idx1-ubyte'),
            test_images_filepath = join(data, 't10k-images-idx3-ubyte/t10k-images-idx3-ubyte'),
            test_labels_filepath = join(data, 't10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte'))

    def __init__(self, training_images_filepath, training_labels_filepath,
                 test_images_filepath, test_labels_filepath):
        self.training_images_filepath = training_images_filepath
        self.training_labels_filepath = training_labels_filepath
        self.test_images_filepath = test_images_filepath
        self.test_labels_filepath = test_labels_filepath

    def read_images_labels(self, images_filepath, labels_filepath):
        '''
        Used to read one set of images and labels, either training or test

        Parameters:
            images_filepath   Full pathname for images
            labels_filepath   Full pathname for labels

        Returns:  images, labels

        If either file is missing, program will exit
        '''
        try:
            labels = []
            with open(labels_filepath, 'rb') as file:
                magic, size = struct.unpack(">II", file.read(8))
                if magic != 2049:
                    raise ValueError(f'Magic number mismatch, expected 2049, got {magic}')
                labels = array("B", file.read())

            with open(images_filepath, 'rb') as file:
                magic, size1, rows, cols = struct.unpack(">IIII", file.read(16))
                if magic != 2051:
                    raise ValueError(f'Magic number mismatch, expected 2051, got {magic}')
                image_data = array("B", file.read())
                assert size1 == size
                assert rows == 28
                assert cols == 28

            images = []
            for i in range(size):
                images.append([0] * rows * cols)
            for i in range(size):
                img = np.array(image_data[i * rows * cols:(i + 1) * rows * cols])
                img = img.reshape(rows, cols)
                images[i][:] = img

            return images, labels
        except FileNotFoundError as e:
            print (f'Could not find file {e.filename}')
            exit(1)
        except ValueError as e:
            print (e)
            exit(1)


    def load_data(self):
        '''
        Load training and test data

        Returns:  (x_train, y_train), (x_test, y_test)

        '''
        x_train, y_train = self.read_images_labels(self.training_images_filepath, self.training_labels_filepath)
        x_test, y_test = self.read_images_labels(self.test_images_filepath, self.test_labels_filepath)
        return (x_train, y_train), (x_test, y_test)


def parse_args():
    parser = ArgumentParser(__doc__)
    parser.add_argument('--show', default=False, action='store_true', help='Controls whether plot will be displayed')
    parser.add_argument('--figs', default='./figs', help='Location for storing plot files')
    parser.add_argument('--cmap',default='Blues',help='Colour map')
    parser.add_argument('--resize', default=None,type=int,nargs='+',help='Used to resize image')
    parser.add_argument('--m', '-m', default=12,type=int,help='Number of rows for display')
    parser.add_argument('--seed', default=None, type=int, help='For initializing random number generator')
    return parser.parse_args()


def create_mask(mask_file=None,data='../data',size=28):
    '''
    Create a mask by reading from a file, if a name is provided.
    If there is no mask file, return all ones.

    Parameters:
        mask_file   File name
        data        Location for storing data files
        size        Mask will be size x size pixels

    Returns:
        An array containing the actual mask
        Text showing mask file name
    '''
    if mask_file == None:  return np.ones((size,size)),'no mask'
    mask_path = Path(join(data, mask_file)).with_suffix('.npy')
    product = np.load(mask_path)
    print (f'Loaded mask from {mask_path}')
    return product,f'Mask = {mask_file}'

def columnize(x):
    '''
    Convert list of images into an array of column vectors, one column per image

    Parameters:
         x        List of images
    '''
    x1 = np.array(x)
    _,n_rows,n_cols=x1.shape
    assert n_rows == n_cols
    x_img_no_last = np.transpose(x1,[1,2,0])
    x_columnized_img_no_last = np.reshape(x_img_no_last, (n_rows*n_cols, -1))
    return np.transpose(x_columnized_img_no_last,[1,0])

def create_indices(y, nclasses=10, nimages=1000, rng=np.random.default_rng()):
    '''
    Create list of indices for data  ensuring that there are
    precisely nimages images from each class.

    Parameters:
        y          Vector of labels
        rng        Random number generator
        nclasses   Number of classes
        nimages    Number of images per class

    Returns:
        An array with one column per digit class, one
        row for sequence within class
    '''
    product = np.zeros((nimages, nclasses), dtype=int)
    class_counts = np.zeros((nclasses), dtype=int)
    for k in rng.permutation(len(y)):
        image_class = y[k]
        i = class_counts[image_class]
        if i < nimages:
            product[i, image_class] = k
            class_counts[image_class] += 1
        else:
            if np.min(class_counts) == nimages:
                return product

    raise RuntimeError(f'Failed to find {nimages} labels in {nclasses} classes')

def create_entropies(images,selector,bins=20,m=28):
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

def get_mutual_information(x,y):
    '''
    Calculate mutual information between two vectors. This is a wrapper for
    sklearn.feature_selection.mutual_info_classif, which expects X to be a matrix

    Parameters:
        x     A vector
        y     Another vector
    '''
    return mutual_info_classif(x[:, np.newaxis],y)

if __name__ == '__main__':
    rc('font', **{'family': 'serif',
                  'serif': ['Palatino'],
                  'size': 8})
    rc('text', usetex=True)
    fig = figure(figsize=(8, 12))
    start = time()
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    mnist_dataloader = MnistDataloader.create()
    (x_train,y_train),_ = mnist_dataloader.load_data()

    for i in range(args.m):
        k = rng.choice(len(x_train))
        img = np.array(x_train[k])
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

    elapsed = time() - start
    minutes = int(elapsed/60)
    seconds = elapsed - 60*minutes
    print (f'Elapsed Time {minutes} m {seconds:.2f} s')

    if args.show:
        show()


