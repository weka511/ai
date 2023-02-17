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

'''Solve problem of Figure 2 from Smith et al'''

from numpy          import array,log
from scipy.optimize import minimize
from scipy.special  import xlogy

def get_q(q0):
    return array([q0[0],1-q0[0]])

def F(q0,p=array([0.8,0.2])):
    q = q0/q0.sum()
    return (xlogy(q,q/p)).sum()

result = minimize(F,array([0.5,0.5]),
                  bounds   = [(0,1),(0,1)],
                  callback = lambda x:print(x,F(x)))
print (result.x)
