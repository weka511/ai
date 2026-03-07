# mnist

Exploring use of Active Inference for Pattern Recognition.

The inspiration comes from the following papers by Friston et al:
- [Supervised structure learning](https://arxiv.org/abs/2311.10300)
- [From pixels to planning: scale-free active inference](https://arxiv.org/abs/2407.20292)

File|Description
-------------------------------------|-----------------------------------------------------------------------------------------
build.sh|Shell script to build files in pipeline
gibbs1.py|Testbed for Gibbs sampling
mask.py|Mask out pixels that carry little information
mnist.py|Functions for accessing MNIST data
pipeline.py|Program for creating files needed for each step in analysis pipeline
shared|Folder containing utilities, such as logger and check_for_stopped
style.py|Classes to manange character styles
visualize.py|Visualise MNIST data
