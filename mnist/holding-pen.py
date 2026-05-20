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
    
class DecayFunction(ABC):
    '''
    This class is used to model the decay function of Blei and Frazier
    '''
    @abstractmethod
    def __call__(self,mutual_information,iclass):
        '''
        Apply decay function
        '''
        ...

class NoDecay(DecayFunction):
    '''
    Pass mutual information through unchanged
    '''
    def __call__(self,mutual_information,_):
        return mutual_information    
    
class Window(DecayFunction):
    '''
    This decay function replaces values that are less than the mean by zero;
    values that exceed the cutoff value are truncated to the cutoff.
    
    Attributes:
        lambdas
        exponent
        cutoff_values
        
    '''
    def __init__(self,lambdas,cutoff_multiplier=2,exponent=1):
        self.lambdas = lambdas
        self.exponent = exponent
        self.cutoff_values = cutoff_multiplier/self.lambdas
        
    def __call__(self,mutual_information,iclass):
        mutual_information[mutual_information < 1/self.lambdas[iclass]] = 0
        mutual_information[mutual_information > self.cutoff_values[iclass]] = self.cutoff_values[iclass] 
        return mutual_information**self.exponent
   
    
class EstablishStylesUsingCRP(Stage2):
    '''
    Organize images into styles following Chinese Restaurant Process
    '''
    def __init__(self):
        super().__init__('Establish Styles New','establish-styles-new',needs_output_file=True)
        self.R = 25
        self.C = 40        

    def _load_supplementary_files(self):
        super()._load_supplementary_files()
        file =  (self.data_path / self.args.mutual_information).with_suffix('.npz')  #DODO move
        data = np.load(file)
        self.mutual_information = data['mi'] 
        self.logger.log(f'Loaded {file}')
        
    def _execute(self):
        n_classes0,m,n = self.mutual_information.shape
        assert m == n 
        _, n_classes = self.indices.shape
        assert n_classes == n_classes0
        
        lambdas,calculated_mutual_information = self._fit_exponentials(n_classes=n_classes)
        f = Window(lambdas)
        adapters = [
            StyleAdapter(iclass,
                         clusters=self._gibbs_sample(f(self.mutual_information[iclass,:,:].copy(),iclass),m,
                                                       alpha=1/lambdas[iclass]).clusters,
                         indices=self.indices[:,iclass]) 
            for iclass in range(n_classes)
        ]
            
        self._plot(n_classes,adapters)
        self._plot_mutual_information(lambdas,calculated_mutual_information)
        
    @staticmethod
    def get_off_diagonal(M):
        '''
        Extract the off-diagonal elements of a matrix
        
        Parameters:
            M
            
        Returns:
            A 1 dmenional matrix containing off-diagonal elements
        '''
        M = np.copy(M)
        np.fill_diagonal(M,np.nan)
        M = M.ravel()
        return M[~np.isnan(M)]
    
    @staticmethod
    def _exponential_distribution(x,decay):
        '''
        A function used to fit represent an exponential distribution
        
        Parameters:
            x
            decay     Parameter for distribution
        '''
        return decay*np.exp(-decay*x)
    
    def _fit_exponentials(self,n_classes=10,num=25):
        '''
        A function used to fit an exponential to histogram of mutual information
        
        Parameters:
            n_classes   Number of image classes
            num         Number of bins for histogram
            
        Returns:
            lambdas          Parametefs for mutual information, one per class
            bin_mutual_informationd_points   Mid points of bins generated by np.histogram
        '''
        bins = np.linspace(0,1,num=num+1,endpoint=True)
        lambdas = np.zeros((n_classes))
        for i in range(n_classes):
            M = EstablishStylesUsingCRP.get_off_diagonal(self.mutual_information[i,:,:])
            n,bin_edges = np.histogram(M,bins=bins,density=True)
            bin_mutual_informationd_points = (bin_edges[0:-1] + bin_edges[1:])/2
            popt,pcov = curve_fit(EstablishStylesUsingCRP._exponential_distribution,bin_mutual_informationd_points,n)
            lambdas[i] = popt[0]    
        return lambdas,bin_mutual_informationd_points
    
    def _plot_mutual_information(self,lambdas,calculated_mutual_information,bins=25):
        '''
        Plot distribution of mutual information within each digit class
        
        Parameters:
            lambdas
            calculated_mutual_information
            bins
        '''
        fig = figure(figsize=(12,12))
        fig.suptitle('Mutual Information')
        n_classes,_,_ = self.mutual_information.shape
        for i in range(n_classes):
            ax = fig.add_subplot(3,4,1+i)
            ax.hist(EstablishStylesUsingCRP.get_off_diagonal(self.mutual_information[i,:,:]),
                    bins=bins,density=True)
            ax.plot(calculated_mutual_information,EstablishStylesUsingCRP._exponential_distribution(calculated_mutual_information,lambdas[i]),
                    label=r'$\lambda=$'+f'{lambdas[i]:.3f}')
            ax.axvline(x=1/lambdas[i],label=f'Mean={1/lambdas[i]:.3f}',c='r',ls='dashed')
            ax.set_title(f'Class={i}')
            ax.legend()
            
        fig.tight_layout(pad=2,h_pad=2,w_pad=2)
        fig.savefig((self.figs_path / self.args.out).with_suffix('.png'))
        
    def _gibbs_sample(self,mutual_information,m,alpha=1.0):
        '''
        Perform Gibbs sampling
        
        Parameters:
            mutual_information
            m
        '''
        crp = CRPdd(mutual_information,alpha=alpha,rng=self.rng)

        for j in range(self.args.M*m):
            crp.gibbs(self.rng.choice(m))
            if j % self.args.freq == self.args.freq - 1:
                self.logger.log(f'{j} {crp.changed_allocations}')
                crp.changed_allocations = 0
  
        return crp
    
    def _plot(self,n_classes,adapters): 
        '''
        Display images, organized by styles.
        
        Parameters:
            n_classes
            adapters
        '''
        for i in range(n_classes):  
            fig = figure(figsize=(12, 12))
            fig.suptitle(f'Class={i}')
            for j in range(len(adapters[i])):
                image_generator = adapters[i].generate_images(j)
                for k,image_number in enumerate(image_generator):
                    if j < self.R and k < self.C:
                        ax = fig.add_subplot(self.R, self.C, j*self.C + k + 1)
                        img = self.x[image_number,:].reshape(self.args.size,self.args.size)
                        ax.imshow(img,cmap=self.args.cmap)
                        ax.axis('off')
  
            fig.savefig((self.figs_path / (self.args.out + str(i))).with_suffix('.png'))          