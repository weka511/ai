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
from os.path  import join
from pathlib import Path
import struct
from array import array
import numpy as np
from matplotlib.pyplot import figure, show
from matplotlib import rc,cm
import kagglehub

class MnistDataloader(object): # snarfed from https://www.kaggle.com/code/hojjatk/read-mnist-dataset
    def __init__(self, training_images_filepath,training_labels_filepath,
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
        return (x_train, y_train),(x_test, y_test)

def parse_args():
    parser = ArgumentParser(__doc__)
    parser.add_argument('--show', default=False, action='store_true', help='Controls whether plot will be displayed')
    parser.add_argument('--figs', default='./figs', help='Location for storing plot files')
    return parser.parse_args()

def show_images(images, title_texts):
    cols = 5
    rows = int(len(images)/cols) + 1
    for index,x in enumerate(zip(images, title_texts)):
        image = x[0]
        title_text = x[1]
        plt.subplot(rows, cols, index+1)
        plt.imshow(image, cmap=plt.cm.gray)
        if (title_text != ''):
            plt.title(title_text, fontsize = 15);

def equalize(raw,L=256):
    raw = np.array(raw)
    m,n = raw.shape
    hist,edges = np.histogram(raw,bins=np.arange(0,256))
    v = edges[:-1]
    cdf = np.cumsum(np.append(hist,[0]))
    i = 0
    while cdf[i] == 0: i += 1
    cdf_min = cdf[i]
    h = np.round((L-1)* (cdf-cdf_min)/(m*n-cdf_min))
    result = np.zeros((m,n))
    for i in range(m):
        for j in range(n):
            result[i,j] =h[raw[i,j]]
    return result

if __name__ == '__main__':
    rc('font', **{'family': 'serif',
                  'serif': ['Palatino'],
                  'size': 8})
    rc('text', usetex=True)
    fig = figure(figsize=(24, 12))
    args = parse_args()

    input_path  = kagglehub.dataset_download("hojjatk/mnist-dataset")

    training_images_filepath = join(input_path, 'train-images-idx3-ubyte/train-images-idx3-ubyte')
    training_labels_filepath = join(input_path, 'train-labels-idx1-ubyte/train-labels-idx1-ubyte')
    test_images_filepath = join(input_path, 't10k-images-idx3-ubyte/t10k-images-idx3-ubyte')
    test_labels_filepath = join(input_path, 't10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte')
    mnist_dataloader = MnistDataloader(training_images_filepath, training_labels_filepath, test_images_filepath, test_labels_filepath)

    (x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()
    equalized = equalize(x_train[0])

    ax1 = fig.add_subplot(1,4,1)
    ax1.imshow( x_train[0], cmap=cm.gray)
    ax2 = fig.add_subplot(1,4,2)
    ax2.imshow( equalized, cmap=cm.gray)
    ax3 = fig.add_subplot(1,4,3)
    ax3.hist(x_train[0])
    ax4 = fig.add_subplot(1,4,4)
    ax4.hist(equalized)

    if args.show:
        show()


