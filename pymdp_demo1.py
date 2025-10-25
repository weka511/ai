#!/usr/bin/env python

# Example snarfed from Conor Heins et al, pymdp: A Python library for active inference in discrete state spaces

import numpy as np

from pymdp import utils, maths
from pymdp.agent import Agent
from pymdp.envs import Env


class Custom_Env(Env):
	def __init__(self, rng=np.random.default_rng(None)):
		self.state = 0
		self.rng = rng

	def step(self, action):
		match action:
			case 0:
				self.state = 0 if self.rng.uniform() > 0.5 else 1
			case 1:
				self.state = 2

		match self.state:
			case 0:
				return 0
			case 1:
				return 1
			case 2:
				return self.rng.choice(3)


def create_A(inv_temperature=0.5):
	A = utils.obj_array(1)
	A[0] = np.eye((3), dtype=float)
	A[0][:, 2] = maths.softmax(inv_temperature * A[0][:, 2])
	return A


def create_B():
	B = utils.obj_array(1)
	B[0] = np.zeros((3, 3, 2))
	B[0][:, :, 0] = np.array([[0.5, 0.5, 0.5],
                           [0.5, 0.5, 0.5],
                           [0.0, 0.0, 0.0]])
	B[0][:, :, 1] = np.array([[0.0, 0.0, 0.0],
                           [0.0, 0.0, 0.0],
                           [1.0, 1.0, 1.0]])
	return B


def create_C(n_obs=3):
	C = utils.obj_array_uniform([n_obs])
	return C


def create_D(n_states=3):
	D = utils.obj_array(1)
	D[0] = utils.onehot(1, n_states)
	return D


if __name__ == '__main__':
	my_agent = Agent(A=create_A(),
                  B=create_B(),
                  C=create_C(),
                  D=create_D())
	env = Custom_Env()
	action = 0
	T = 10
	for t in range(T):
		o_t = env.step(action)
		print('o_t', o_t)
		qs = my_agent.infer_states([o_t])
		print('qs', qs)
		my_agent.infer_policies()
		action = my_agent.sample_action()
		print('action', action)
		action = int(action.squeeze())
