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
    Establish styles within classes using mutual information
'''

import numpy as np
from mnist import get_mutual_information

class Style(object):
    '''
    This class assigns images to Styles
    '''
    def __init__(self,exemplar_index):
        self.exemplar_index = exemplar_index
        self.indices = [exemplar_index]

    def __len__(self):
        return len(self.indices)

    def add(self,new_index):
        self.indices.append(new_index)



class StyleList(object):
    '''
    This class manages the collection of all Styles for one digit class
    '''
    @staticmethod
    def build(x,indices,i_class=0,nimages=10,threshold=0.1):
        x_class = x[indices[:,i_class],:]   # All vectors in this digit-class
        product = StyleList(x_class)
        for j in range(nimages):
            matching_style,mi = product.get_best_match(j)
            if matching_style == None or mi < threshold:
                product.add(Style(j))
            else:
                matching_style.add(j)
        return product

    def  __init__(self,x_class):
        self.styles = []
        self.x_class = x_class

    def __len__(self):
        return len(self.styles)

    def add(self,style):
        self.styles.append(style)

    def get_best_match(self,index):
        style_best_match = None
        mi_best_match = -1
        x = self.x_class[index,:]

        for i in range(len(self.styles)):
            candidate_style = self.styles[i]
            y = self.x_class[candidate_style.exemplar_index,:]
            mi_canditate = get_mutual_information(x,y)
            if mi_canditate > mi_best_match:
                mi_best_match = mi_canditate
                style_best_match = candidate_style

        return  style_best_match,mi_best_match

    def save(self,file):
        m = len(self.styles)
        n = max(len(style) for style in self.styles)
        Allocations = -1 * np.ones((m,n),dtype=int)
        for i,style in enumerate(self.styles):
            for j,index in enumerate(style.indices):
                Allocations[i,j] = index
        np.save(file,Allocations)
        print (f'Saved styles in {file}')

