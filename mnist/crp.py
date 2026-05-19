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
    This module implements the Chinese Restaurant Process
'''
from abc import ABC,abstractmethod
import numpy as np

class DecayFunction(ABC):
    '''
    This class is used to model the decay function of Blei and Frazier
    '''
    @abstractmethod
    def __call__(self,mutual_information):
        '''
        Apply decay function
        '''
        ...

class NoDecay(DecayFunction):
    '''
    Pass mutual information through unchanged
    '''
    def __call__(self,mutual_information):
        return mutual_information    
    
    
class ChineseRestaurantProcess:
    
    UNASSIGNED = -1
    
    def __init__(self,mutual_information,
                 f=NoDecay(),rng=np.random.default_rng(),alpha=2.0):
        self.fd = f(1/mutual_information)
        self.rng = rng
        self.m,self.n = mutual_information.shape
        assert self.m == self.n
        self.rng = rng
        self.alpha = alpha
        self.links = np.full((self.m),ChineseRestaurantProcess.UNASSIGNED, dtype=int)
        
    def build(self):
        indices = self.rng.permutation(self.m)
        for i in range(self.m):
            self.links[indices[i]] = self.rng.choice(indices[:i+1],p=self._get_p_init(i))
    
    def gibbs(self):
        for i in range(self.m):
            self.links[i] = self.rng.choice(self.m,p=self._get_p(i))
    
    def _get_p_init(self,i):
        p = 1/self.fd[i,:]
        p[i] = self.alpha
        p = p[:i+1]
        return p / p.sum()
            
    def _get_p(self,i):
        p = 1/self.fd[i,:]
        p[i] = self.alpha
        return p/p.sum() 
    
def main():
    crp = ChineseRestaurantProcess(np.ones((12,12)))
    crp.build()

if __name__ == '__main__':
    main()