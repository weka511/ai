#!/usr/bin/env python

# Copyright (C) 2025 Simon Crase  simon@greenweaves.nz

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

'''Template for Python code'''


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


class MnistDataloader(object): # snarfed from https://www.kaggle.com/code/hojjatk/read-mnist-dataset
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


def show_images(images, title_texts):
    cols = 5
    rows = int(len(images) / cols) + 1
    for index, x in enumerate(zip(images, title_texts)):
        image = x[0]
        title_text = x[1]
        plt.subplot(rows, cols, index + 1)
        plt.imshow(image, cmap=plt.cm.gray)
        if (title_text != ''):
            plt.title(title_text, fontsize=15)


def equalize(raw, L=256):
    '''
    Histogram equalization after wikipedia

    https://en.wikipedia.org/wiki/Histogram_equalization
    '''
    def create_histogram(pixels):
        Count = {}
        for value in np.nditer(pixels):
            pixel_value = int(value)
            if not pixel_value in Count:
                Count[pixel_value] = 0
            Count[pixel_value] += 1
        return Count

    def create_Cdf(Count):
        NonZeros = sorted([Count.keys()])
        Cdf = {}
        running_total = 0
        for key, count in Count.items():
            running_total += count
            Cdf[key] = running_total
        return Cdf

    def equalize_cdf(Cdf, mn):
        h = {}
        for key, count in Cdf.items():
            h[key] = int((L - 1) * (count - Cdf[0]) / (mn - Cdf[0]) + 0.5)
        return h

    pixels = np.array(raw)
    m, n = pixels.shape
    h = equalize_cdf(create_Cdf(create_histogram(pixels)), m * n)
    return np.fromfunction(np.vectorize(lambda i, j: h[pixels[i, j]]), pixels.shape, dtype=int)


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
    fig = figure(figsize=(24, 12))
    args = parse_args()

    input_path = kagglehub.dataset_download("hojjatk/mnist-dataset")

    training_images_filepath = join(input_path, 'train-images-idx3-ubyte/train-images-idx3-ubyte')
    training_labels_filepath = join(input_path, 'train-labels-idx1-ubyte/train-labels-idx1-ubyte')
    test_images_filepath = join(input_path, 't10k-images-idx3-ubyte/t10k-images-idx3-ubyte')
    test_labels_filepath = join(input_path, 't10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte')
    mnist_dataloader = MnistDataloader(training_images_filepath, training_labels_filepath, test_images_filepath, test_labels_filepath)

    (x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()
    equalized = equalize_hist(np.array(x_train[0]))
    equalized0 = equalize(np.array(x_train[0]))

    img = resize(np.array(x_train[0]),(32,32))
    ax11 = fig.add_subplot(2, 4, 1)
    ax11.imshow(img, cmap=cm.gray)
    ax11.set_title('Raw')
    ax21 = fig.add_subplot(2, 4, 5)
    ax21.hist(img)

    ax13 = fig.add_subplot(2, 4, 3)
    ax13.imshow(histeq(img), cmap=cm.gray)
    ax13.set_title('janeriksolem')
    ax23 = fig.add_subplot(2, 4, 7)
    ax23.hist(histeq(img))

    ax14 = fig.add_subplot(2, 4, 4)
    ax14.imshow(equalize_hist(img), cmap=cm.gray)
    ax14.set_title('Skimage')
    ax24 = fig.add_subplot(2, 4, 8)
    ax24.hist(equalize_hist(img))

    fig.savefig('Equalize')

    if args.show:
        show()


