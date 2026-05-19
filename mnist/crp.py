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
    
class Table:
    seq = -1
    tables = []
    
    def __init__(self):
        self.indices = []
        Table.seq += 1
        self.seq = Table.seq
        Table.tables.append(self)
     
    def __len__(self):
        return len(self.indices)
    
    def __str__(self):
        return f'Table {self.seq}: {self.indices}'
    
    def append(self,index):
        self.indices.append(index)
    

class ChineseRestaurantProcess:
    '''
    This class represents a distance dependent Chinese Restaurant Process. The
    distance is a monotonically decreasing funtion of mutual information
    '''
    
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
        self.tables = np.empty((self.m),dtype=Table)
        
    def build(self):
        indices = self.rng.permutation(self.m)
        for i in range(self.m):
            current = indices[i]
            p = self._get_p_init(i)
            link_to = self.rng.choice(indices[:i+1],p=p)
            self.links[current] = link_to
            table = Table() if current == link_to else self.tables[link_to]
            self.tables[current] = table
            table.append(current)
        for table in Table.tables:
            print (table)
   
    
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