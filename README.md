# Active Inference

Code written to understand Karl Friston on Active Inference

## Programs based on [Parr, Pezzulo, Friston 2022 Textbook](https://direct.mit.edu/books/oa-monograph/5299/Active-InferenceThe-Free-Energy-Principle-in-Mind)

File|Description
-------------------|---------------------------------------------------------------------------------------------------
hmm.py|Hidden Markove Model Example from Section 7-2
maze.py|Example from Section 7.3 Decision Making and Planning as Inference

## Exploring use of Active Inference for Pattern Recognition.

The inspiration comes from the following papers by Friston et al:
- [Supervised structure learning](https://arxiv.org/abs/2311.10300)
- [From pixels to planning: scale-free active inference](https://arxiv.org/abs/2407.20292)

File|Description
-------------------------------------|-----------------------------------------------------------------------------------------
cluster.py|Cluster Analysis for MNIST: establish variability of mutual information within and between classes
eda.py|Exploratory Data Analysis for MNIST
eda_mi.py|Exploratory Data Analysis for MNIST: figure out variability of mutual information within and between classes
establish_styles.py|Establish styles within classes using mutual information
establish_subset.py|Extract subsets of MNIST to facilitate replication
establish_most_informative_pixels.py|Determine which pixels are most relevant to classifying images
mnist.py|Functions for accessing MNIST data


## Miscellaneous

File  | Remarks |
---------------|-------------------------------------------------------------------------------------------
ai.bib|Bibliography
ai.tex|Derivations of equations
ai.py| Common code for my Active Inference project
ai.wpr|Project for Python Code
template.py|Template for creating additional programs

## Subfolders

Folder|Description
-------|---------------------------------------------------------------------------------------------------
bogacz|Code based on[A tutorial on the free-energy framework for modelling perception and learning, by Rafal Bogacz](https://www.sciencedirect.com/science/article/pii/S0022249615000759)
pymdp_demos|Various pydmp demos
step_by_step|[A Step-by-Step Tutorial on Active Inference and its Application to Empirical Data](https://www.researchgate.net/publication/348153427_A_Step-by-Step_Tutorial_on_Active_Inference_and_its_Application_to_Empirical_Data)
lawki|[Life as we Know it](https://royalsocietypublishing.org/doi/10.1098/rsif.2013.0475)

## The code has been developed and tested using the following versions of Python and its libraries.

Library|Version
-----------|-------------------
python|3.14.0
numpy|2.3.4
matplotlib|3.10.7
pymdp|0.0.7.1
seaborn|0.13.2
