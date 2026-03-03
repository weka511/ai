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
    This class serves as a container for methods that manipulate an array used to mask
    out pixels that carry little information
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
        if mask_file == None:
            report ('No mask specified')
            return Mask(np.ones((size,size))),'no mask',[]
        data_path = Path(data).resolve()
        mask_path = (data_path / mask_file).with_suffix('.npz')
        mask_data = np.load(mask_path)
        product = Mask(mask_data['mask'])
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
        n = len(selector)
        def create_1d_images():
            '''
            Standardize images to be mxm, equalize, then convert to 1d
            '''
            m0,n0 = images[0].shape
            product = np.zeros((n, m*m))
            for i in selector:
                if m == m0 and m == n0:
                    standard_image = images[i]
                else:
                    standard_image = resize(np.array(images[i]),(m,m))
                # img = equalize_hist(standard_image) Issue #61
                product[i] = np.reshape(standard_image,-1) # Issue #61
            return product

        def create_entropies_from_1d_images(images1d):
            '''
            Calculate probability density for each pixel, then calculate entropy
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
    def build(x_train,indices,bins='doane',m=28,fraction=0.5):
        entropies = Mask.create_entropies(x_train[indices],list(range(len(indices))),bins=bins,m=m)
        mu = np.mean(entropies)
        sigma = np.std(entropies)
        img = np.reshape(entropies,(m,m))
        threshold = mu - fraction*sigma
        pixels = Mask.cull(entropies,threshold=threshold).reshape(m,m)
        return FatMask(pixels,img,entropies,mu,threshold)

    def __init__(self,pixels):
        self.pixels = pixels.copy()
        self.pixels1d = pixels.reshape(-1)

    def save(self,file,bins):
        np.savez(file, mask=self.pixels,bins=bins)

    def apply(self,x):
        return np.multiply(x, self.pixels1d)

    def __getitem__(self,i):
        return self.pixels1d[i]

    def get_ratio(self):
        return self.pixels1d.sum()/len(self.pixels1d)

class FatMask(Mask):
    '''
    Used when we first create a mask to hold additional attributes
    that are not needed by later stages in pipeline
    '''
    def __init__(self,pixels,img,entropies,mu,threshold):
        super().__init__(pixels)
        self.img = img
        self.entropies = entropies
        self.mu = mu
        self.threshold = threshold

