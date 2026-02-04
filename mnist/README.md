# mnist

Exploring use of Active Inference for Pattern Recognition.

The inspiration comes from the following papers by Friston et al:
- [Supervised structure learning](https://arxiv.org/abs/2311.10300)
- [From pixels to planning: scale-free active inference](https://arxiv.org/abs/2407.20292)

File|Description
-------------------------------------|-----------------------------------------------------------------------------------------
cluster.py|Cluster Analysis for MNIST: establish variability of mutual information within and between classes
display_styles|Display representatives of all styles created by establish_styles.py
eda.py|Exploratory Data Analysis for MNIST
eda_mi.py|Exploratory Data Analysis for MNIST: figure out variability of mutual information within and between classes
establish_styles.py|Establish styles within classes using mutual information
establish_subset.py|Extract subsets of MNIST to facilitate replication
establish_most_informative_pixels.py|Determine which pixels are most relevant to classifying images
mnist.py|Functions for accessing MNIST data
ssl.py|
style.py|Classes to manange character styles
