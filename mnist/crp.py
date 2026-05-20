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
from pathlib import Path
import numpy as np
from shared.utils import Logger

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
        self.links = {}
        Table.seq += 1
        self.seq = Table.seq
        Table.tables.append(self)
     
    def __len__(self):
        return len(self.indices)
    
    def __str__(self):
        return f'Table {self.seq}: {self.indices}'
    
    def append(self,index):
        self.indices.append(index)
        
    def link(self,start,end):
        self.links[start] =  end
        
    #def remove(self,index):
        #self.indices.remove(index)
        #self.links.pop(index)
    
    SELF_LINK = 0
    CHAIN = 1
    CYCLE = 2
        
    def search(self,start):
        history = []
        current = start
        if self.links[current] == current: return Table.SELF_LINK,current,[]
        while True:
            next_link = self.links[current]
            if next_link == current: return Table.CHAIN,next_link,[current]
            if next_link in history: return Table.CYCLE,next_link,history
            history.append(next_link)
            current = next_link
            
        
class ChineseRestaurantProcess:
    '''
    This class represents a distance dependent Chinese Restaurant Process. The
    distance is a monotonically decreasing funtion of mutual information
    '''
    
    UNASSIGNED = -1
    
    def __init__(self,mutual_information,
                 f=NoDecay(),rng=np.random.default_rng(),
                 alpha=2.0,logger=None):
        self.fd = f(1/mutual_information)
        self.rng = rng
        self.m,self.n = mutual_information.shape
        assert self.m == self.n
        self.rng = rng
        self.alpha = alpha
        self.links = np.full((self.m),ChineseRestaurantProcess.UNASSIGNED, dtype=int)
        self.tables = np.empty((self.m),dtype=Table)
        self.logger = logger
        
    def build(self):
        indices = self.rng.permutation(self.m)
        for i in range(self.m):
            current = indices[i]
            link_to = self.rng.choice(indices[:i+1],p=self._get_p_init(i))
            self.links[current] = link_to
            table = Table() if current == link_to else self.tables[link_to]
            self.tables[current] = table
            table.append(current)
            table.link(current,link_to)
        for table in Table.tables:
            self._log(table)
   
    
    def gibbs(self):
        for current in range(self.m):
            table = self.tables[current]
            self._log(table)
            chain_type,running,history = table.search(current)
            link_to = self.rng.choice(self.m,p=self._get_p(current))
            if table.links[current] == link_to:
                self._log(f'Node {current}: linking back to {table.links[current]}')
            else:
                self._log(f'Node {current}: was linked to {table.links[current]}, now linked  to {link_to}')
                #table.remove(current)
                self._log(table)
                self.links[current] = link_to
    
    def _get_p_init(self,i):
        p = 1/self.fd[i,:]
        p[i] = self.alpha
        p = p[:i+1]
        return p / p.sum()
            
    def _get_p(self,current):
        p = 1/self.fd[current,:]
        p[current] = self.alpha
        return p/p.sum() 
    
    def _log(self,s):
        self.logger.log(s)
    
    
def main():
    with Logger(Path(__file__).stem,path='./logs') as logger:
        crp = ChineseRestaurantProcess(np.ones((12,12)),logger=logger)
        crp.build()
        crp.gibbs()

if __name__ == '__main__':
    main()