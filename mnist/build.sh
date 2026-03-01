#!/bin/bash
set -euxo pipefail
./pipeline.py establish-subsets --nimages $1 -o subsets$1
./pipeline.py establish-mask --indices subsets$1 -o mask$1
./pipeline.py establish-styles --indices subsets$1 --mask mask$1 --o styles$1
./pipeline.py establish-likelihoods --indices subsets$1 --mask mask$1 --styles styles$1 -o likelihoods$1
./pipeline.py recognize-digits --indices subsets$1 --mask mask$1 --styles styles$1 --likelihoods likelihoods$1 -o results$1 --N 100
