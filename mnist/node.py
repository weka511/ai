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
    Allow digit classes to be linked togther to support Gibbs sampling
'''

from unittest import TestCase, main
import numpy as np

class Node:
    NOT_LINKED = 1
    
    def __init__(self,seq):
        self.seq = seq
        self.link_to = Node.NOT_LINKED
        self.links_from = []
        
    def __str__(self):
        return f'{self.seq}->{self.link_to}'    

class NodeSet:
    
    @staticmethod
    def build(n,rng=np.random.default_rng()):
        Product = NodeSet(n)
        for i in range(n):
            Product.link(i,rng.choice(i+1))

        return Product
    
    def __init__(self,n):
        self.nodes = []
        for i in range(n):
            self.nodes.append(Node(i))
            
    def __len__(self):
        return len(self.nodes)
    
    def __getitem__(self,index):
        return self.nodes[index]
    
    def link(self,index_from,index_to):
        assert self.nodes[index_from].link_to == Node.NOT_LINKED
        self.nodes[index_from].link_to = index_to
        self.nodes[index_to].links_from.append(index_from)
        
    def break_link(self,index_from,index_to):
        self.nodes[index_from].link_to = Node.NOT_LINKED
        self.nodes[index_to].links_from.remove(index_from)
        
    def dfs(self,index):
        result = []
        for i in self.nodes[index].links_from:
            result.append(i)
            result += self.dfs(i)
        return result
    
    def nodes(self):
        for node in self.nodes:
            yield node
            
class TestNode(TestCase):
    def setUp(self):
        self.network = NodeSet(10)
        self.assertEqual(10,len(self.network))
        self.network.link(0,1)  
        self.network.link(1,0)
        self.network.link(2,1)        
        self.network.link(3,0)
        self.network.link(4,4)
        self.network.link(5,2)
        self.network.link(6,3)  
        self.network.link(7,6)  
        self.network.link(8,6)  
        self.network.link(9,3)
        
    def test_link(self):
        network = NodeSet(10)
        self.assertEqual(10,len(network))     
        network.link(3,0)
        self.assertEqual(0,network[3].link_to)
        self.assertIn(3,network[0].links_from)        
        
    def test_break(self):
        self.network.break_link(3,0)
        self.assertEqual(Node.NOT_LINKED,self.network[3].link_to)
        self.assertNotIn(3,self.network[0].links_from)
        self.assertIn(1,self.network[0].links_from)
        
    def test_bfs(self):
        self.assertEqual(set([6,7,8,9]), set(self.network.dfs(3)))
        
    def test_build(self):
        nodeset = NodeSet.build(41)
        for node in nodeset:
            self.assertLessEqual(node.link_to,node.seq)
        
if __name__ == '__main__':
    main()
    