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
    Determine which pixels are most relevant to classifying images
'''

from argparse import ArgumentParser
from os.path import join
from pathlib import Path
from time import time
from matplotlib.pyplot import figure, show
from matplotlib import rc
import numpy as np
from scipy.stats import entropy
from pymdp.maths import softmax, spm_log_single as log_stable
from skimage.exposure import equalize_hist
from skimage.transform import resize
from mnist import MnistDataloader

def parse_args():
    parser = ArgumentParser(__doc__)
    parser.add_argument('--show', default=False, action='store_true', help='Controls whether plot will be displayed')
    parser.add_argument('--figs', default='./figs', help='Location for storing plot files')
    parser.add_argument('--data', default='./data', help='Location for storing data files')
    parser.add_argument('--indices', default='establish_subset.npy', help='Location for storing data files')
    parser.add_argument('--bins', default=17, type=int, help='Number of bins for histogram')
    return parser.parse_args()

def create_entropies(images,selector,bins=20,m=32):
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
        product = np.zeros((n, m*m))
        for i in selector:
            img = equalize_hist(resize(np.array(images[i]),(m,m)))
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

def show_image(entropies,ax=None,fig=None,cmap='Blues'):
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

def show_culled(entropies,n,mu,sigma,min0,ax=None,cmap='Blues'):
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
    culled_img = np.copy(entropies)
    culled_img[culled_img < mu + n*sigma] = min0
    ax.imshow(culled_img,cmap=cmap)
    ax.set_title(rf'Culled {abs(n)}$\sigma$ below $\mu$' if n != 0 else r'Culled all below $\mu$')

def show_histogram(entropies,mu,sigma,ax=None):
    '''
    Show histogram of entropies.

    Parameters:
        entropies  An array of entropies, with one entry for each pixel
        mu         Mean entropy
        sigma      Standard deviation for entropy
        ax         Axis for displaying data
    '''
    ax.hist(img.reshape(-1),bins='fd',density=True,color='xkcd:blue',label='Histogram')
    ax.set_xlabel('H')
    ax.set_ylabel('Frequency')
    ax.set_title('Entropy of pixels')
    ax.axvline(mu,c='xkcd:red',ls='-',label=r'$\mu$')
    ax.axvline(mu-sigma,c='xkcd:red',ls='--',label=r'$\mu-\sigma$')
    ax.axvline(mu-2.0*sigma,c='xkcd:red',ls=':',label=r'$\mu-2\sigma$')
    ax.legend()

if __name__ == '__main__':
    rc('font', **{'family': 'serif',
                  'serif': ['Palatino'],
                  'size': 8})
    rc('text', usetex=True)
    fig = figure(figsize=(12, 12))
    start = time()
    args = parse_args()
    indices = np.load(join(args.data,args.indices)).astype(int)

    mnist_dataloader = MnistDataloader.create(data=args.data)
    (x_train, _), _ = mnist_dataloader.load_data()
    x_train = np.array(x_train)
    entropies = create_entropies(x_train[indices],list(range(len(indices))),bins=args.bins)
    mu = np.mean(entropies)
    sigma = np.std(entropies)
    min0 = np.min(entropies)
    img = np.reshape(entropies,(32,32))
    cmap='seismic'
    show_image(img,ax=fig.add_subplot(2,3,1),fig=fig,cmap=cmap)
    show_culled(img,-2,mu,sigma,min0,ax = fig.add_subplot(2,3,2),cmap=cmap)
    show_culled(img,-1,mu,sigma,min0,ax = fig.add_subplot(2,3,3),cmap=cmap)
    show_culled(img,-0.5,mu,sigma,min0,ax = fig.add_subplot(2,3,4),cmap=cmap)
    show_culled(img,0,mu,sigma,min0,ax = fig.add_subplot(2,3,5),cmap=cmap)
    show_histogram(img,mu,sigma,ax=fig.add_subplot(2,3,6))

    fig.suptitle(r'Information based on \emph{' + f'{args.indices}' + ',} ' + f'{len(indices//10):,d} images per class, {args.bins} bins')
    fig.savefig(join(args.figs,Path(__file__).stem))
    elapsed = time() - start
    minutes = int(elapsed/60)
    seconds = elapsed - 60*minutes
    print (f'Elapsed Time {minutes} m {seconds:.2f} s')

    if args.show:
        show()
