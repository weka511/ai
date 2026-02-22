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

from os.path import join
from pathlib import Path
import struct
from array import array
from unittest import TestCase,main
import numpy as np
from scipy.stats import entropy
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


    def load_data(self,verbose=True):
        '''
        Load training and test data

        Returns:  (x_train, y_train), (x_test, y_test)
        '''
        x_train, y_train = self.read_images_labels(self.training_images_filepath, self.training_labels_filepath)
        if verbose:
            print (f'Loaded training data from {self.training_images_filepath},')
            print (f'labels from {self.training_labels_filepath}')
        x_test, y_test = self.read_images_labels(self.test_images_filepath, self.test_labels_filepath)
        if verbose:
            print (f'Loaded test data from {self.test_images_filepath},')
            print (f'labels from {self.test_labels_filepath}')
        return (x_train, y_train), (x_test, y_test)

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
    if mask_file == None:
        print ('No mask specified')
        return np.ones((size,size)),'no mask',[],[]
    mask_path = Path(join(data, mask_file)).with_suffix('.npz')
    mask_data = np.load(mask_path)
    product = mask_data['mask']
    bins = mask_data['bins']
    n = mask_data['n']
    print (f'Loaded mask from {mask_path}')
    return product,f'Mask = {mask_file}',n,bins

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

class TestSequence(TestCase):

    def setUp(self):
        mnist_dataloader = MnistDataloader.create()
        (self.x_train,self.y_train),_ = mnist_dataloader.load_data(verbose=False)


    def test_labels_match(self):
        '''
        Test that records are retrieved consistently. Data were generated by executing:
            rng = np.random.default_rng()
            for i in rng.choice(len(self.y_train),size=100):
                print (f'self.assertEqual({self.y_train[i]},self.y_train[{i}])')
        '''
        self.assertEqual(2,self.y_train[45648])
        self.assertEqual(7,self.y_train[25115])
        self.assertEqual(8,self.y_train[22150])
        self.assertEqual(5,self.y_train[26771])
        self.assertEqual(3,self.y_train[14361])
        self.assertEqual(3,self.y_train[58793])
        self.assertEqual(6,self.y_train[54621])
        self.assertEqual(1,self.y_train[44745])
        self.assertEqual(6,self.y_train[37859])
        self.assertEqual(3,self.y_train[5186])
        self.assertEqual(4,self.y_train[7734])
        self.assertEqual(3,self.y_train[26448])
        self.assertEqual(7,self.y_train[5000])
        self.assertEqual(0,self.y_train[ 5498])
        self.assertEqual(8,self.y_train[ 35264])
        self.assertEqual(1,self.y_train[ 3758])
        self.assertEqual(8,self.y_train[ 12944])
        self.assertEqual(8,self.y_train[ 5534])
        self.assertEqual(3,self.y_train[ 31283])
        self.assertEqual(8,self.y_train[ 13423])
        self.assertEqual(7,self.y_train[ 24907])
        self.assertEqual(8,self.y_train[ 45399])
        self.assertEqual(1,self.y_train[ 57430])
        self.assertEqual(5,self.y_train[ 25002])
        self.assertEqual(3,self.y_train[ 20213])
        self.assertEqual(9,self.y_train[20164])
        self.assertEqual(6,self.y_train[41782])
        self.assertEqual(3,self.y_train[43482])
        self.assertEqual(4,self.y_train[42750])
        self.assertEqual(0,self.y_train[51680])
        self.assertEqual(9,self.y_train[33199])
        self.assertEqual(2,self.y_train[56403])
        self.assertEqual(7,self.y_train[7565])
        self.assertEqual(9,self.y_train[44645])
        self.assertEqual(0,self.y_train[58617])
        self.assertEqual(1,self.y_train[53273])
        self.assertEqual(7,self.y_train[37984])
        self.assertEqual(0,self.y_train[22978])
        self.assertEqual(2,self.y_train[16925])
        self.assertEqual(0,self.y_train[14840])
        self.assertEqual(0,self.y_train[20747])
        self.assertEqual(3,self.y_train[10116])
        self.assertEqual(4,self.y_train[45184])
        self.assertEqual(5,self.y_train[13551])
        self.assertEqual(2,self.y_train[47862])
        self.assertEqual(3,self.y_train[53875])
        self.assertEqual(2,self.y_train[39161])
        self.assertEqual(5,self.y_train[52875])
        self.assertEqual(2,self.y_train[8886])
        self.assertEqual(5,self.y_train[32750])
        self.assertEqual(4,self.y_train[46172])
        self.assertEqual(5,self.y_train[34050])
        self.assertEqual(7,self.y_train[41769])
        self.assertEqual(0,self.y_train[49639])
        self.assertEqual(8,self.y_train[56913])
        self.assertEqual(5,self.y_train[30112])
        self.assertEqual(8,self.y_train[25986])
        self.assertEqual(1,self.y_train[54979])
        self.assertEqual(2,self.y_train[9962])
        self.assertEqual(6,self.y_train[32736])
        self.assertEqual(5,self.y_train[41486])
        self.assertEqual(5,self.y_train[9800])
        self.assertEqual(4,self.y_train[32558])
        self.assertEqual(3,self.y_train[48817])
        self.assertEqual(8,self.y_train[47632])
        self.assertEqual(7,self.y_train[16990])
        self.assertEqual(1,self.y_train[3212])
        self.assertEqual(3,self.y_train[9610])
        self.assertEqual(9,self.y_train[4663])
        self.assertEqual(3,self.y_train[15246])
        self.assertEqual(8,self.y_train[2302])
        self.assertEqual(4,self.y_train[20702])
        self.assertEqual(8,self.y_train[19351])
        self.assertEqual(4,self.y_train[17590])
        self.assertEqual(0,self.y_train[14693])
        self.assertEqual(4,self.y_train[9976])
        self.assertEqual(6,self.y_train[23774])
        self.assertEqual(7,self.y_train[30462])
        self.assertEqual(7,self.y_train[19636])
        self.assertEqual(9,self.y_train[47853])
        self.assertEqual(6,self.y_train[56918])
        self.assertEqual(0,self.y_train[34847])
        self.assertEqual(2,self.y_train[32438])
        self.assertEqual(3,self.y_train[47461])
        self.assertEqual(2,self.y_train[3165])
        self.assertEqual(1,self.y_train[22868])
        self.assertEqual(3,self.y_train[57194])
        self.assertEqual(1,self.y_train[10863])
        self.assertEqual(8,self.y_train[38324])
        self.assertEqual(8,self.y_train[47136])
        self.assertEqual(2,self.y_train[21537])
        self.assertEqual(0,self.y_train[39194])
        self.assertEqual(1,self.y_train[39957])
        self.assertEqual(3,self.y_train[21055])
        self.assertEqual(4,self.y_train[39396])
        self.assertEqual(9,self.y_train[33597])
        self.assertEqual(1,self.y_train[45955])
        self.assertEqual(3,self.y_train[107])
        self.assertEqual(1,self.y_train[28857])
        self.assertEqual(4,self.y_train[27914])
        self.assertEqual(1,self.y_train[40716])
        self.assertEqual(3,self.y_train[28171])
        self.assertEqual(7,self.y_train[2618])
        self.assertEqual(6,self.y_train[38996])
        self.assertEqual(6,self.y_train[11610])
        self.assertEqual(7,self.y_train[27714])
        self.assertEqual(3,self.y_train[26644])
        self.assertEqual(4,self.y_train[30151])
        self.assertEqual(0,self.y_train[57700])
        self.assertEqual(4,self.y_train[45025])
        self.assertEqual(5,self.y_train[18829])
        self.assertEqual(9,self.y_train[13111])
        self.assertEqual(5,self.y_train[18839])
        self.assertEqual(8,self.y_train[52237])
        self.assertEqual(1,self.y_train[32530])
        self.assertEqual(7,self.y_train[27406])
        self.assertEqual(4,self.y_train[26494])
        self.assertEqual(3,self.y_train[2669])
        self.assertEqual(3,self.y_train[13670])
        self.assertEqual(7,self.y_train[35196])
        self.assertEqual(0,self.y_train[35050])
        self.assertEqual(2,self.y_train[17123])
        self.assertEqual(5,self.y_train[25557])
        self.assertEqual(1,self.y_train[32295])
        self.assertEqual(7,self.y_train[37055])


if __name__ == '__main__':
    main()
