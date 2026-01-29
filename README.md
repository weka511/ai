# Active Inference

Code written to understand Karl Friston on Active Inference

## Programs based on [Parr, Pezzulo, Friston 2022 Textbook](https://direct.mit.edu/books/oa-monograph/5299/Active-InferenceThe-Free-Energy-Principle-in-Mind)


File|Description
-------------------|---------------------------------------------------------------------------------------------------
hmm.py|Hidden Markove Model Example from Section 7-2
maze.py|Example from Section 7.3 Decision Making and Planning as Inference

## Exploring use of Active Inference for Pattern Recognition. The inspiriation comes from the following papers by Friston et al:
- [Supervised structure learning](https://arxiv.org/abs/2311.10300)
- [From pixels to planning: scale-free active inference](https://arxiv.org/abs/2407.20292)

File|Description
-------------------------------------|-----------------------------------------------------------------------------------------
establish_most_informative_pixels.py|Determine which pixels are most relevant to classifying images
mnist.py|Functions for accessing MNIST data

## Programs based on [Life as we Know it](https://royalsocietypublishing.org/doi/10.1098/rsif.2013.0475)

File|Description
-------------------|---------------------------------------------------------------------------------------------------
ai.bib|Bibliography
ai.tex|Derivations of equations
selforg.py|Replication of Figure 2, Self Organization and the emergence of macroscopic behaviour, from Friston & Ao, [Free Energy, Value, and Attractor](https://www.hindawi.com/journals/cmmm/2012/937860/)
huygens.py|Test for Huyghens oscillator in selforg.py

## Programs based on [A Step-by-Step Tutorial on Active Inference and its Application to Empirical Data](https://www.researchgate.net/publication/348153427_A_Step-by-Step_Tutorial_on_Active_Inference_and_its_Application_to_Empirical_Data)

File|Description
-------------------|---------------------------------------------------------------------------------------------------
example2.py|Example 2
exercise2.py|Exercise 2
figure2.py|Figure 2
message_passing.py|Ported from [Message_passing_example.m](https://github.com/rssmith33/Active-Inference-Tutorial-Scripts/blob/main/Message_passing_example.m)
pomdp.py|POMDP example and library for solvers
pomdp_driver.py|POMDP example

## Programs based on [A tutorial on the free-energy framework for modelling perception and learning, by Rafal Bogacz](https://www.sciencedirect.com/science/article/pii/S0022249615000759)

 File  | Remarks |
---------------|-------------------------------------------------------------------------------------------
feex1.py| Exercise 1--posterior probabilities
feex2.py| Exercise 2--most likely size
feex3.py| Exercise 3--neural implementation
feex5.py| Exercise 5--learn variance

## ODE solvers

File  | Remarks |
---------------|-------------------------------------------------------------------------------------------
euler.py|Euler's method
rk4.py|Workhorse Runge-Kutta 4th order
sde.py|Euler-Maruyama method

## pydmp demos

File  | Remarks |
--------------------|-------------------------------------------------------------------------------------------
pymdp_demo1.py|[Conor Heins et al--pymdp: A Python library for active inference in discrete state spaces](https://arxiv.org/abs/2201.03904)
tutorial1.py|[Active inference from scratch](https://pymdp-rtd.readthedocs.io/en/latest/notebooks/active_inference_from_scratch.html)
tutorial2.py|[The Agent API](https://pymdp-rtd.readthedocs.io/en/latest/notebooks/using_the_agent_class.html)
tutorial_common.py|Common code for tutorials
tutorial_maze.py|[Active Inference Demo: T-Maze Environment](https://pymdp-rtd.readthedocs.io/en/latest/notebooks/tmaze_demo.html)

## Miscellaneous

File  | Remarks |
---------------|-------------------------------------------------------------------------------------------
ai.py| Common code for my Active Inference project
ai.wpr|Project for Python Code
template.py|Template for creating additional programs

## The code has been developed and tested using the following versions of Python and its libraries.

Library|Version
-----------|-------------------
python|3.14.0
numpy|2.3.4
matplotlib|3.10.7
pymdp|0.0.7.1
seaborn|0.13.2
