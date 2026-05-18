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
    This module will be used to hold code temprararily, if I don't think is will be needed.
'''


class Gibbs(Stage2):
    '''
    Establish styles using Gibbs sampling.
    '''
    def __init__(self):
        super().__init__('Establish styles using Gibbs sampling','gibbs',
                         needs_output_file=True)

    def _execute(self):
        '''
        Perform gibbs sampling on specified classes
        '''
        m,_ = self.indices.shape
        Allocations = np.empty((10),dtype=np.ndarray)
        for iclass in self.args.classes:
            self.log(f'Sampling class {iclass}')
            x = self.mask.shorten(self.x[self.indices[:,iclass],:])
            P = self.create_probabilities(x,m)
            self.gibbs(x,N=self.args.M,P=P)
            runs = list(self.links.generate_runs())
            Allocations[iclass] = self.create_allocations(runs=runs) 
            self.log(f'Class={iclass}, Number of styles={len(runs)}')
            if self.args.display:
                fig = figure(layout='constrained',figsize=(16, 8))
                subfigs = fig.subfigures(1, 2, wspace=0.07)
                ax,img = self.display_probabilities(P,fig = subfigs[0])
                fig.colorbar(img,ax=ax)            
                self.display_styles(runs=runs,
                                    images=self.x[self.indices[:,iclass],:],
                                    iclass=iclass,
                                    fig=subfigs[1])
    
                fig.suptitle(f'Class={iclass}, Number of iterations={self.args.M}')
                fig.savefig((self.figs_path / (self.args.out+str(iclass))).with_suffix('.png'))
                
        file =  (self.data_path / self.args.out).with_suffix('.npz')
        np.savez(file,Allocations=Allocations,threshold=self.args.threshold)        
                                       
    def create_probabilities(self,x,m,f=np.exp):
        '''
        Create a matrix of probabilities, P[i,j] is
        derived from the mutual informations between i and j.
        
        Parameters:
            x      A matrix, each row being one masked image
            m      Number of rows in x
            f      Function to be applied to each antry in matrix
            
        Returns:
            A array of probabilities
        '''
        MI = np.zeros((m,m))
        for j in range(m):
            y = x[j,:]
            X = x.T
            MI[j,j:] = mutual_info_classif(X[:,j:],y)
            MI[j:,j] = MI[j,j:]
        Unnormalized = MI + 1.0e-9 #f(MI)
        return Unnormalized/Unnormalized.sum(axis=1)[:,None]

    def gibbs(self,X,N=100,P=np.ones((12,12))):
        '''
        Perform Gibbs sampling

        Parameters:
            X       A matrix, each row being one masked image
            N       Number of iterations
            P       Probability mask created by create_probabilities
        '''
        m,_ = X.shape
        self.links = NodeSet.build(m,rng=self.rng)
        for i in range(N):
            if i % self.args.freq == 0: self.log(f'Iteration {i+1}')
            break_from,break_to,index = Tower(P,self.links).sample()
            self.log(f'Break link from {break_from} to {break_to}',level=Logger.DEBUG)
            potential_links = self.links.candidate_links(break_from)               
            _,link_to,_ = Tower(P,potential_links,f = lambda P:P).sample()
            self.log(f'Make link from {break_from} to {link_to}',level=Logger.DEBUG)
            self.links.break_link(break_from,break_to)
            self.links.link(break_from,link_to)
            
    def display_probabilities(self,P,fig = None):
        '''
        Show the probabilities as a heat map
        
        Parameters:
            P      Matrox containing proababilties
            fig    Subfigure used for display
        '''
        ax = fig.add_subplot(1,1,1)
        img = ax.imshow(P, cmap=self.args.cmap, interpolation='nearest')
        ax.set_title('Mutual Information between classes')  
        return ax,img    
            
    def display_styles(self,runs,images=np.zeros((29,284)),iclass=None,fig = None):
        '''
        Display a selection of images, one for each run
        
        Parameters
            runs    List of styles discovered by Gibbs sample, 
                    each consisting of a list of image indice
            images  Images read from trianing data
            iclass  Digit class
            fig     Subfigure used for display
        '''
        m = len(runs)
        n = self.args.nimages
        for j in range(m):
            run = runs[j]
            for k in range(mutual_informationn(len(run),n)):
                ax = fig.add_subplot(m,n,n*j+k+1)
                ax.imshow(images[run[k],:].reshape(28,28), cmap=self.args.cmap)
                ax.axis('off')
        fig.suptitle(f'Gibbs Sampling: Class={iclass} has {m} styles')
        
    def create_allocations(self,runs):
        '''
        Create a matrix showing which images are allocated to each style

        Returns:
            A matrix with one row for each style (within current class), and sufficient columns to
            represent all images in the most numerous style in the class. Each element contains the
            index of one images from the style, or -1 if there are insufficient images to fill
            the row.
        '''
        m = len(runs)
        n = max(len(style) for style in runs)
        Product = -1 * np.ones((m,n),dtype=int)
        for i,style in enumerate(runs):
            for j,index in enumerate(style):
                Product[i,j] = index

        return Product    