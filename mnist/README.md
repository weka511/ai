# mnist

Exploring use of Active Inference for Pattern Recognition.

The inspiration comes from the following papers by Friston et al:
- [Supervised structure learning](https://arxiv.org/abs/2311.10300)
- [From pixels to planning: scale-free active inference](https://arxiv.org/abs/2407.20292)

File|Description
-------------------------------------|-----------------------------------------------------------------------------------------
build.sh|Shell script to build files in pipeline
command.py|Command interpreter shared by pipeline.py and visualize.py
crp.dd|Chinese Restaurant Process
holding-pen.py|This module will be used to hold code temprararily, if I don't think is will be needed.
mask.py|Mask out pixels that carry little information
mnist.py|Functions for accessing MNIST data
node.py|Allow digit classes to be linked togther to support Gibbs sampling--see Blei & Frazier [Distance Dependent Chinese Restaurant Processes](https://www.jmlr.org/papers/volume12/blei11a/blei11a.pdf)
pipeline.py|Program for creating files needed for each step in analysis pipeline
shared|Folder containing utilities, such as logger and check_for_stopped
style.py|Classes to manage character styles
visualize.py|Visualise MNIST data
