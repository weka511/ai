#!/bin/bash

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

set -euxo pipefail
if [ $# -gt 0 ];  then
	./pipeline.py establish-subsets --nimages $1 -o subsets$1 
	./pipeline.py establish-mask --indices subsets$1 -o mask$1
	./pipeline.py establish-styles --indices subsets$1 --mask mask$1 --o styles$1
	./pipeline.py establish-likelihoods --indices subsets$1 --mask mask$1 --styles styles$1 -o likelihoods$1
	if [ $# -gt 1 ];  then
		./pipeline.py recognize-digits --indices subsets$1 --mask mask$1 --styles styles$1 --likelihoods likelihoods$1 -o results$1 --N $2
	  else
		./pipeline.py recognize-digits --indices subsets$1 --mask mask$1 --styles styles$1 --likelihoods likelihoods$1 -o results$1 --N 100
	fi
  else
    echo "No arguments supplied"
fi

