#!/usr/bin/env python

#   Copyright (C) 2024 Simon Crase

#   This program is free software: you can redistribute it and/or modify
#   it under the terms of the GNU General Public License as published by
#   the Free Software Foundation, either version 3 of the License, or
#   (at your option) any later version.

#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.

#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.

'''Test for Huyghens oscillator in selforg.py'''


from argparse import ArgumentParser
from os.path import basename, splitext, join
from time import time
import numpy as np
from matplotlib.pyplot import figure, show
from euler import euler
from selforg import Huyghens


def parse_args(tFinal=10.0,
               Nt=5000,
               V0=1.0):
    '''Define and parse command line arguments'''
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--tFinal', type=float, default=tFinal, help=f'Final time for integration [{tFinal}]')
    parser.add_argument('--Nt', type=int, default=Nt, help=f'Number of time steps [{Nt}]')
    parser.add_argument('--V0', type=float, default=V0, help=f'Initial Velocity [{V0}]')
    parser.add_argument('--show', default=False, action='store_true', help='Show plots')
    return parser.parse_args()


def get_name_for_save(extra=None,
                      sep='-',
                      figs='./figs'):
    '''
    Extract name for saving figure

    Parameters:
        extra    Used if we want to save more than one figure to distinguish file names
        sep      Used if we want to save more than one figure to separate extra from basic file name
        figs     Path name for saving figure

    Returns:
        A file name composed of pathname for figures, plus the base name for
        source file, with extra distinguising information if required
    '''
    basic = splitext(basename(__file__))[0]
    name = basic if extra == None else f'{basic}{sep}{extra}'
    return join(figs, name)


if __name__ == '__main__':
    start = time()
    args = parse_args()
    huyghens = Huyghens()

    tArray = np.linspace(0.0, args.tFinal, args.Nt)
    ssp0 = np.array([0.0, args.V0], float)
    sspSolution = euler(huyghens.Velocity, ssp0, tArray)

    xSolution = sspSolution[:, 0]
    vSolution = sspSolution[:, 1]

    fig = figure(figsize=(6, 6))
    ax = fig.add_subplot(2, 2, 1)
    ax.plot(tArray, xSolution)
    ax.set_ylabel('x(t)')

    ax = fig.add_subplot(2, 2, 2)
    ax.plot(tArray, vSolution)
    ax.set_xlabel('t(s)')
    ax.set_ylabel('v(t)')

    ax = fig.add_subplot(2, 2, 3)
    ax.plot(xSolution, vSolution)
    ax.set_xlabel('x(t)')
    ax.set_ylabel('v(t)')
    fig.suptitle(f'tFinal={args.tFinal}, Nt={args.Nt}, V0={args.V0}')
    fig.savefig(get_name_for_save())
    elapsed = time() - start
    minutes = int(elapsed / 60)
    seconds = elapsed - 60 * minutes
    print(f'Elapsed Time {minutes} m {seconds:.2f} s')
    if args.show:
        show()
