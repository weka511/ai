#!/usr/bin/env python

# Copyright (C) 2023 Simon Crase

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

'''Solve problem of Exercise 2 from Smith et al'''

from example2 import step1
import numpy as np

if __name__=='__main__':
    D = np.array([0.5, 0.5])

    A = np.array([[0.9, 0.1],
               [0.1, 0.9]])

    B = np.array([[1,0],
              [1,0]])

    o1 = np.array([1,0])

    o2 = np.array([0,1])

    s1 = np.array([0.5, 0.5])

    s2 = np.array([0.5, 0.5])

    print (step1(D,B,A,o=[o1],s=[s1,s2],log=lambda x:np.log(x+0.01)))
