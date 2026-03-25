#!/usr/bin/env python

from jax import numpy as jnp, random as jr
from pymdp import utils
from pymdp.agent import Agent

key = jr.PRNGKey(0)
keys = jr.split(key, 3)

num_obs = [3, 5]
num_states = [3, 2]
num_controls = [3, 1]

A = utils.random_A_array(keys[0], num_obs, num_states)
B = utils.random_B_array(keys[1], num_states, num_controls)
C = utils.list_array_uniform([[no] for no in num_obs])

agent = Agent(A=A, B=B, C=C, batch_size=1)

# Discrete observation indices for each modality
obs = [jnp.array([1]), jnp.array([2])]

# Use agent.D as the initial empirical prior
qs, info = agent.infer_states(obs, empirical_prior=agent.D, return_info=True)
# Optional diagnostic: current variational free energy for each batch element.
vfe = info["vfe"]
q_pi, neg_efe = agent.infer_policies(qs)

sample_keys = jr.split(keys[2], agent.batch_size + 1)