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
    This module is the command interpreter for pipeline.py and visualize.py.
'''
from abc import ABC,abstractmethod
from time import time, strftime,localtime
from pathlib import Path
from matplotlib.pyplot import show
import numpy as np
from mnist import MnistDataloader, MnistException
from shared.utils import Logger,create_xkcd_colours
from mask import Mask

class CommandException(RuntimeError):
    '''
    Allows Command to raise exceptions
    '''
    def __init__(self,message):
        super().__init__(message)

class Command(ABC):
    '''
    Parent class for procesing requests. Each command exports a string that can
    be used by the user to execute the desired functionality.
    
    Attributes:
        description        Used for documentation only
        name               Name used to schedule command
        needs_output_file  Indicates that the user needs to provide an output file
        colours            Each digit class has its own colour
        log                Logger
    '''
    commands = {}

    @staticmethod
    def build(command_list):
        '''
        Load a list of commands that will be available for execution
        '''
        for command in command_list:
            Command.commands[command.name] = command

    @staticmethod
    def get_names():
        '''
        Used to construct command line argument
        '''
        return [name for name in Command.commands.keys()]

    @staticmethod
    def execute_one(args):
        '''
        Execute one command that has been selected by user. This
        encapsulates code that is shared with visualize.py

        Parameters:
            args      Comand line arguments
        '''
        start = time()
        with Logger(Path(__file__).stem,path=args.logs) as logger:
            command = Command.commands[args.command]
            command.set_args(args)
            command.set_logger(logger)
            code = 0
            try:
                command.execute()
            except FileNotFoundError as e:
                command.log(f'Error: {e.filename} not found.',level=Logger.ERROR)
                code = 1
            except MnistException as e:
                command.log(f'Mnist Exception {e}',level=Logger.ERROR)
                code = 1
            except CommandException as e:
                command.log(f'Command Exception {e}',level=Logger.ERROR)
                code = 1
            except RuntimeError as e:
                command.log(f'RuntimeError {e}',level=Logger.ERROR)
                code = 1                
            finally:
                elapsed = time() - start
                mutual_informationnutes = int(elapsed / 60)
                seconds = elapsed - 60 * mutual_informationnutes
                logger.log(f'Elapsed Time {mutual_informationnutes} m {seconds:.2f} s')
                if code > 0: exit(code)

            if args.show:
                show()

    def __init__(self,description,name,
                 needs_output_file=False,n=10):
        '''
        Parameters:
            description        Used for documentation only
            name               Name used to schedule command
            needs_output_file  Indicates that the user needs to provide an output file
            n                  Number of colours that will be needed, one for each digit class
        '''
        self.description = description
        self.name = name
        self.needs_output_file = needs_output_file
        self.colours = create_xkcd_colours(n)

    def get_description(self):
        '''
        Used for documentation only
        '''
        return self.description

    def set_args(self,args):
        '''
        Store command line arguments
        '''
        self.args = args
        self.rng = np.random.default_rng(args.seed)

    def execute(self):
        '''
        Shared code for executing command:
        - Load mnist images and other data used by commands
        - apply mask
        '''
        self.log (f'{self.get_description()} {strftime("%a, %d %b %Y %H:%M:%S +0000", localtime())}')

        if self.needs_output_file and self.args.out == None:
            raise CommandException ('Output file must be specified')

        self.data_path = Path(self.args.data).resolve()
        self.figs_path = Path(self.args.figs).resolve()

        self.load_mnist_data()
        self._load_supplementary_files()

        self._execute()   # Perform actual command

    def load_mnist_data(self):
        '''
        Load training and test data. Download data if it is not present.
        '''
        dataloader = MnistDataloader.create(data=self.args.data,report = lambda x:self.log(x))
        try:
            (self.x_train, self.y_train), (self.x_test, self.y_test), = dataloader.load_data()
        except MnistException:
            dataloader.download_data(data=self.args.data)
            (self.x_train, self.y_train), (self.x_test, self.y_test), = dataloader.load_data()
        self.x = MnistDataloader.columnize(self.x_train)

    def _load_supplementary_files(self):
        '''
        Load additional files needed by some commands
        '''
        pass

    @abstractmethod
    def _execute(self):
        '''
        Execute command: must be implemented for each class
        '''
        ...

    def set_logger(self,logger):
        '''
        Attach a logger to Command
        '''
        self.logger = logger

    def log(self,message,level=Logger.INFO):
        '''
        Log messages
        '''
        self.logger.log(message,level=level)

    def digitize_images(self,imgs,equalize=False):
        '''
        Used by EstablishLikelihoods and RecognizeDigits to prepare images
        '''
        equalized_images = equalize_hist(imgs) if equalize else imgs
        return np.digitize(equalized_images,self.bins)

class Stage1(Command):
    '''
    This is the parent class for commands that depend on an index file
    
    Attributes:
        description        Used for documentation only
        name               Name used to schedule command
        needs_output_file  Indicates that the user needs to provide an output file
        colours            Each digit class has its own colour
        indices            Each entry is the index of one image in data
        nimages            Number of images
        log                Logger
    '''
    def _load_supplementary_files(self):
        '''
        Load index files
        '''
        super()._load_supplementary_files()
        file =  (self.data_path / self.args.indices).with_suffix('.npz')
        index_data = np.load(file)
        self.indices = index_data['indices']
        self.nimages = index_data['nimages']
        self.log (f'Loaded indices from {file} for {self.nimages} images')

class Stage2(Stage1):
    '''
    This is the parent class for commands that depend on an index file and a mask
    
    Attributes:
        description        Used for documentation only
        name               Name used to schedule command
        needs_output_file  Indicates that the user needs to provide an output file
        colours            Each digit class has its own colour
        indices            Each entry is the index of one image in data
        nimages            Number of images
        mask               Mask used to exclude pixels that contribute little information
        x                  Masked data
        log                Loggers
    '''
    def _load_supplementary_files(self):
        '''
        Load index files and mask
        '''        
        super()._load_supplementary_files()
        self.mask, self.mask_text,self.bins = Mask.create(mask_file=self.args.mask,
                                                          data=self.args.data,
                                                          size=self.args.size,
                                                          report = lambda x:self.log(x))
        self.log('Bins: ' +str(self.bins),level=Logger.DEBUG)
        self.x = self.mask.apply(self.x)


class Stage3(Stage2):
    '''
    This is the parent class for commands that depend on an index file, a mask,  and a style file
    
    Attributes:
        description        Used for documentation only
        name               Name used to schedule command
        needs_output_file  Indicates that the user needs to provide an output file
        colours            Each digit class has its own colour
        indices            Each entry is the index of one image in data
        nimages            Number of images
        mask               Mask used to exclude pixels that contribute little information
        x                  Masked data
        log                Loggers
        Allocation         Allocation of images to styles
        Threshold          Images are included in the same style if mutual information exceeds threshold
    '''
    def _load_supplementary_files(self):
        '''
        Load index files, mask, and styles
        '''           
        super()._load_supplementary_files()
        file =  (self.data_path / self.args.styles).with_suffix('.npz')
        style_data = np.load(file,allow_pickle=True)
        self.Allocations = style_data['Allocations']
        self.threshold = style_data['threshold']
        self.log (f'Loaded Allocations from {file}')

class Stage4(Stage3):
    '''
    This is the parent class for commands that depend on an
    index file, mask, style file, and a likelihoods file
    
    Attributes:
        description        Used for documentation only
        name               Name used to schedule command
        needs_output_file  Indicates that the user needs to provide an output file
        colours            Each digit class has its own colour
        indices            Each entry is the index of one image in data
        nimages            Number of images
        mask               Mask used to exclude pixels that contribute little information
        x                  Masked data
        log                Loggers
        Allocation         Allocation of images to styles
        Threshold          Images are included in the same style if mutual information exceeds threshold
        A                  Likelihoods
    '''
    def _load_supplementary_files(self):
        '''
        Load index files, mask, styles, and Likelihoods
        '''  
        super()._load_supplementary_files()
        file =  (self.data_path / self.args.likelihoods).with_suffix('.npz')
        loaded_data = np.load(file,allow_pickle=True)
        self.class_styles = loaded_data['class_styles']
        self.A = loaded_data['A']
        self.log (f'Loaded Likelihoods from {file}')