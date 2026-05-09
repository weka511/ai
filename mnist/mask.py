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
    This module serves as a container for methods that manipulate an array used to mask
    out pixels that carry little information
'''

from pathlib import Path
import numpy as np
from scipy.stats import entropy
from skimage.transform import resize

class Mask:
    '''
    This class provides methods to manipulate an array used to mask
    out pixels that carry little information
    
    Attributes:
        pixels     An array that indicates whether each pixel should be used
        pixels1d   The same array, reduced to 1 dimension
        entropies  Entropy of each pixel, claulated over all images
        threshold  Mask was calculated by dropping each pixel whose entropy was less than threshold
    '''
    @staticmethod
    def create(mask_file=None,data='../data',size=28,report=print):
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
            The edges of bins that were used to create histogram for mask
        '''
        if mask_file == None:                # TODO check whether this is ever used
            report ('No mask specified')
            return Mask(np.ones((size,size)),None),'no mask',[]
        data_path = Path(data).resolve()
        mask_path = (data_path / mask_file).with_suffix('.npz')
        mask_data = np.load(mask_path)
        product = Mask(mask_data['mask'],
                       entropies=mask_data['entropies'],threshold=mask_data['threshold'])
        bins = mask_data['bins']
        report (f'Loaded mask from {mask_path}')
        return product,f'Mask = {mask_file}',bins

    @staticmethod
    def create_entropies(images,selector,bins=20,m=28):
        '''
        Used to determine which pixels have the most information

        Parameters:
            images     Raw images from NIST
            selector   Indices of images that need to be included
            bins       Number of bins
            m          We will standardize images to be mxm
        '''
        def create_1d_images():
            '''
            Standardize images to be mxm and convert to 1d
            '''
            m0,n0 = images[0].shape
            product = np.zeros((len(selector), m*m))
            for i in selector:
                product[i] = np.reshape(images[i]
                                            if (m == m0 and m == n0)
                                            else resize(np.array(images[i]),(m,m)),-1)
            return product

        def create_entropies_from_1d_images(images1d):
            '''
            Calculate probability density for each pixel, then calculate entropy

            Parameters:
                images1d    Images, each converted to 1d
            '''
            product = np.zeros((m*m))
            for i in range((m*m)):
                hist,edges = np.histogram(images1d[:,i],bins=bins,density=True)
                pdf = hist/np.sum(hist)
                product[i] = entropy(pdf)
            return product

        return create_entropies_from_1d_images(create_1d_images())

    @staticmethod
    def cull(entropies,threshold=0.5):
        '''
        Cull data for display

        Parameters:
            entropies    An array of entropies, with one entry for each pixel
            threshold    Threshold for culling
        '''
        product = np.zeros_like(entropies,dtype=int)
        product[entropies > threshold] = 1
        return product

    @staticmethod
    def build(x_train,indices,bins='doane',m=28,fraction=0.0):
        '''
        Construct mask from a collection of images

        Parameters:
            x_train    Images from mnist
            indices    Identifies which imges are part of collection
            bins       Used by histogram
            m          Images are mxm
            fraction   Used to decide which pixels should be ignored - number of sigmas below mean

        Returns:
            Newly created Mask
        '''
        entropies = Mask.create_entropies(x_train[indices],list(range(len(indices))),bins=bins,m=m)
        mu = np.mean(entropies)
        sigma = np.std(entropies)
        threshold = mu - fraction*sigma
        pixels = Mask.cull(entropies,threshold=threshold).reshape(m,m)
        return Mask(pixels,entropies,threshold)

    def __init__(self,pixels,entropies=[],threshold=0.5):
        '''
        Parameters:
            pixels
        '''
        self.pixels = pixels.astype(int)
        self.m,self.n = pixels.shape
        self.pixels1d = pixels.reshape(-1)
        self.threshold = threshold
        self.entropies = entropies

    def get_mu(self):
        '''
        Get mean entropy
        '''
        return np.mean(self.entropies)
    
    def get_img(self):
        '''
        Get image of entripies for display
        '''
        return np.reshape(self.entropies,(self.m,self.m))

    def save(self,file,bins):
        '''
        Save data needed to recreate mask
        '''
        np.savez(file, 
                 m=self.m,
                 n=self.n,
                 mask=self.pixels,
                 bins=bins,
                 threshold=self.threshold,
                 entropies=self.entropies)

    def apply(self,x):
        '''
        Apply mask to a collection of pixels
        '''
        return np.multiply(x, self.pixels1d)

    def __getitem__(self,i):
        '''
        Allow access to 1 dimensional pixels
        '''
        return self.pixels1d[i]

    def get_ratio(self):
        '''
        Calculate the fraction of pixels that are preserved 
        '''
        return self.pixels1d.sum()/len(self.pixels1d)

    def shorten(self,x):
        '''
        Shorten each image in a collection by removing pixels that would be masked out

        Parameters:
            x       A collection of images
        '''
        return x[:,self.pixels1d > 0]
