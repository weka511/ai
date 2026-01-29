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

'''Read and display MNIST data'''

from argparse import ArgumentParser
from os.path import join
from pathlib import Path
import struct
from array import array
import numpy as np
from matplotlib.pyplot import figure, show
from matplotlib import rc, cm
from skimage.exposure import equalize_hist
from skimage.transform import resize
import kagglehub


class MnistDataloader(object):
    '''
    Read MNIST data

    snarfed from https://www.kaggle.com/code/hojjatk/read-mnist-dataset
    '''
    @staticmethod
    def create(data = './data'):
        training_images_filepath = join(data, 'train-images-idx3-ubyte/train-images-idx3-ubyte')
        training_labels_filepath = join(data, 'train-labels-idx1-ubyte/train-labels-idx1-ubyte')
        test_images_filepath = join(data, 't10k-images-idx3-ubyte/t10k-images-idx3-ubyte')
        test_labels_filepath = join(data, 't10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte')
        return MnistDataloader(training_images_filepath, training_labels_filepath, test_images_filepath, test_labels_filepath)

    def __init__(self, training_images_filepath, training_labels_filepath,
                 test_images_filepath, test_labels_filepath):
        self.training_images_filepath = training_images_filepath
        self.training_labels_filepath = training_labels_filepath
        self.test_images_filepath = test_images_filepath
        self.test_labels_filepath = test_labels_filepath

    def read_images_labels(self, images_filepath, labels_filepath):
        labels = []
        with open(labels_filepath, 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic))
            labels = array("B", file.read())

        with open(images_filepath, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError('Magic number mismatch, expected 2051, got {}'.format(magic))
            image_data = array("B", file.read())
        images = []
        for i in range(size):
            images.append([0] * rows * cols)
        for i in range(size):
            img = np.array(image_data[i * rows * cols:(i + 1) * rows * cols])
            img = img.reshape(28, 28)
            images[i][:] = img

        return images, labels

    def load_data(self):
        x_train, y_train = self.read_images_labels(self.training_images_filepath, self.training_labels_filepath)
        x_test, y_test = self.read_images_labels(self.test_images_filepath, self.test_labels_filepath)
        return (x_train, y_train), (x_test, y_test)


def parse_args():
    parser = ArgumentParser(__doc__)
    parser.add_argument('--show', default=False, action='store_true', help='Controls whether plot will be displayed')
    parser.add_argument('--figs', default='./figs', help='Location for storing plot files')
    return parser.parse_args()




def histeq(im, nbr_bins=256):
    '''
    Histogram equalization after https://www.janeriksolem.net/histogram-equalization-with-python-and.html
    '''
    imhist, bins = np.histogram(im.flatten(), nbr_bins)
    cdf = np.cumsum(imhist)
    cdf = (nbr_bins - 1) * cdf / cdf[-1]
    return np.interp(im.flatten(), bins[:-1], cdf).reshape(im.shape)


if __name__ == '__main__':
    rc('font', **{'family': 'serif',
                  'serif': ['Palatino'],
                  'size': 8})
    rc('text', usetex=True)
    fig = figure(figsize=(8, 12))
    args = parse_args()
    rng = np.random.default_rng()

    mnist_dataloader = MnistDataloader.create()

    (x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()
    m = 8
    for i in range(m):
        k = rng.choice(len(x_train))
        img = resize(np.array(x_train[k]),(32,32))

        ax1 = fig.add_subplot(m, 4, 4*i+1)
        ax1.axis('off')
        ax1.imshow(img, cmap=cm.gray)
        ax2 = fig.add_subplot(m, 4, 4*i+2)
        ax2.hist(img)

        ax3 = fig.add_subplot(m, 4, 4*i+3)
        ax3.imshow(equalize_hist(img), cmap=cm.gray)
        ax1.axis('off')
        ax4 = fig.add_subplot(m, 4, 4*i+4)
        ax4.hist(equalize_hist(img))

        fig.tight_layout(pad=3,h_pad=3,w_pad=3)
    fig.savefig(join(args.figs,'Equalize'))

    if args.show:
        show()


