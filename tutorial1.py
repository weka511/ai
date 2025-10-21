#!/usr/bin/env python

# Copyright (C) 2025 Simon Crase  simon@greenweaves.nz

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

'''
    Tutorial 1: Active inference from scratch
    https://pymdp-rtd.readthedocs.io/en/latest/notebooks/active_inference_from_scratch.html
'''


from argparse import ArgumentParser
from itertools import product
from os.path import join
from pathlib import Path
from warnings import warn
import numpy as np
from matplotlib import rc
from matplotlib.pyplot import figure, show
import seaborn as sns
from pymdp import utils
from pymdp.maths import softmax, spm_log_single as log_stable
from pymdp.control import construct_policies

class AxisIterator:
    '''
    This class creates subplots as needed
    '''
    def __init__(self,figsize=(12,12), n_rows = 3, n_columns = 3,figs='figs',title = ''):
        self.figsize=figsize
        self.n_rows = n_rows
        self.n_columns = n_columns
        self.seq = 0
        self.title = title
        self.figs = figs

    def __iter__(self):
        return self

    def __next__(self):
        '''
        Used to supply subplots
        '''
        if self.seq < self.n_rows*self.n_columns:
            self.seq += 1
        else:
            warn('Too many subplots')

        return self.fig.add_subplot(self.n_rows,self.n_columns,self.seq)

    def __enter__(self):
        rc('font',**{'family' : 'serif',
                     'serif' : ['Palatino'],
                     'size' : 8})
        rc('text', usetex=True)
        self.fig = figure(figsize=self.figsize)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.fig.suptitle(self.title, fontsize=10)
        self.fig.tight_layout(pad=1)
        self.fig.savefig(join(self.figs,Path(__file__).stem))

def plot_likelihood(matrix, xlabels = list(range(9)), ylabels = list(range(9)), title_str = 'Likelihood distribution (A)',ax=None):
    '''
    Plots a 2-D likelihood matrix as a heatmap
    '''

    if not np.isclose(matrix.sum(axis=0), 1.0).all():
        raise ValueError('Distribution not column-normalized! Please normalize (ensure matrix.sum(axis=0) == 1.0 for all columns)')

    sns.heatmap(matrix, xticklabels = xlabels, yticklabels = ylabels, cmap = 'gray', cbar = False, vmin = 0.0, vmax = 1.0,ax=ax)
    ax.set_title(title_str, fontsize=8)
    ax.tick_params(axis='x', labelsize=8)
    ax.tick_params(axis='y', labelsize=8)

def plot_grid(grid_locations, num_x = 3, num_y = 3,ax=None ):
    '''
    Plots the spatial coordinates of GridWorld as a heatmap, with each (X, Y) coordinate
    labeled with its linear index (its `state id`)
    '''
    grid_heatmap = np.zeros((num_x, num_y))
    for linear_idx, location in enumerate(grid_locations):
        y, x = location
        grid_heatmap[y, x] = linear_idx
    sns.set(font_scale=1.5)
    sns.heatmap(grid_heatmap, annot=True, cbar = False, fmt='.0f', cmap='crest',ax=ax)
    ax.tick_params(axis='x', labelsize=8)
    ax.tick_params(axis='y', labelsize=8)

def plot_point_on_grid(state_vector, grid_locations,ax=None,title='Current location'):
    '''
    Plots the current location of the agent on the grid world
    '''
    state_index = np.nonzero(state_vector)[0][0]
    y, x = grid_locations[state_index]
    grid_heatmap = np.zeros((3,3))
    grid_heatmap[y,x] = 1.0
    sns.heatmap(grid_heatmap, cbar = False, fmt='.0f',ax=ax)
    ax.set_title(title, fontsize=8)
    ax.tick_params(axis='x', labelsize=8)
    ax.tick_params(axis='y', labelsize=8)

def plot_beliefs(belief_dist, title_str='',ax=None):
    '''
    Plot a categorical distribution or belief distribution, stored in the 1-D numpy vector `belief_dist`
    '''
    if not np.isclose(belief_dist.sum(), 1.0):
        raise ValueError('Distribution not normalized! Please normalize')

    ax.grid(zorder=0)
    ax.bar(range(belief_dist.shape[0]), belief_dist, color='r', zorder=3)
    ax.set_xticks(range(belief_dist.shape[0]))
    ax.set_title(title_str, fontsize=8)
    ax.tick_params(axis='x', labelsize=8)
    ax.tick_params(axis='y', labelsize=8)

def parse_args():
    parser = ArgumentParser(__doc__)
    parser.add_argument('--seed',type=int,default=None,help='Seed for random number generator')
    parser.add_argument('--show', default=False, action='store_true', help='Controls whether plot will be displayed')
    parser.add_argument('--figs', default='./figs',                   help = 'Location for storing plot files')
    return parser.parse_args()

def create_B_matrix( actions = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'STAY']):
    B = np.zeros( (len(grid_locations), len(grid_locations), len(actions)) )
    for action_id, action_label in enumerate(actions):
        for curr_state, grid_location in enumerate(grid_locations):
            y, x = grid_location
            match action_label:
                case 'UP':
                    next_y = y - 1 if y > 0 else y
                    next_x = x
                case  'DOWN':
                    next_y = y + 1 if y < 2 else y
                    next_x = x
                case 'LEFT':
                    next_x = x - 1 if x > 0 else x
                    next_y = y
                case 'RIGHT':
                    next_x = x + 1 if x < 2 else x
                    next_y = y
                case  'STAY':
                    next_x = x
                    next_y = y
            new_location = (next_y, next_x)
            next_state = grid_locations.index(new_location)
            B[next_state, curr_state, action_id] = 1.0

    return B,actions

def infer_states(observation_index, A, prior):

    '''
    Implement inference here -- NOTE: prior is already passed in, so you don't need to do anything with the B matrix.
    This function has already been given P(s_t). The conditional expectation that creates 'today's prior',
    using 'yesterday's posterior', will happen *before calling* this function

    Returns:    qs
    '''

    log_likelihood = log_stable(A[observation_index,:])
    log_prior = log_stable(prior)
    return softmax(log_likelihood + log_prior)

def get_expected_states(B, qs_current, action):
    '''
    Compute the expected states one step into the future, given a particular action

    '''
    return B[:,:,action].dot(qs_current) # qs_u

def get_expected_observations(A, qs_u):
    '''
    Compute the expected observations one step into the future, given a particular action
    '''

    return  A.dot(qs_u) # qo_u

def entropy(A):
    ''' Compute the entropy of a set of conditional distributions, i.e. one entropy value per column '''

    H_A = - (A * log_stable(A)).sum(axis=0)

    return H_A

def kl_divergence(qo_u, C):
    ''' Compute the Kullback-Leibler divergence between two 1-D categorical distributions'''

    return (log_stable(qo_u) - log_stable(C)).dot(qo_u)

def calculate_G(A, B, C, qs_current, actions):

    G = np.zeros(len(actions)) # vector of expected free energies, one per action

    H_A = entropy(A) # entropy of the observation model, P(o|s)

    for action_i in range(len(actions)):

        qs_u = get_expected_states(B, qs_current, action_i) # expected states, under the action we're currently looping over
        qo_u = get_expected_observations(A, qs_u)           # expected observations, under the action we're currently looping over

        pred_uncertainty = H_A.dot(qs_u) # predicted uncertainty, i.e. expected entropy of the A matrix
        pred_div = kl_divergence(qo_u, C) # predicted divergence

        G[action_i] = pred_uncertainty + pred_div # sum them together to get expected free energy

    return G

class GridWorldEnv():

    def __init__(self,starting_state = (0,0)):

        self.init_state = starting_state
        self.current_state = self.init_state
        print(f'Starting state is {starting_state}')

    def step(self,action_label):
        (Y, X) = self.current_state

        match action_label:
            case 'UP':
                Y_new = Y - 1 if Y > 0 else Y
                X_new = X
            case 'DOWN':
                Y_new = Y + 1 if Y < 2 else Y
                X_new = X
            case 'LEFT':
                Y_new = Y
                X_new = X - 1 if X > 0 else X
            case 'RIGHT':
                Y_new = Y
                X_new = X +1 if X < 2 else X
            case 'STAY':
                Y_new, X_new = Y, X

        self.current_state = (Y_new, X_new) # store the new grid location

        obs = self.current_state # agent always directly observes the grid location they're in

        return obs

    def reset(self):
        self.current_state = self.init_state
        print(f'Re-initialized location to {self.init_state}')
        obs = self.current_state
        print(f'..and sampled observation {obs}')

        return obs

def calculate_G_policies(A, B, C, qs_current, policies):

    G = np.zeros(len(policies)) # initialize the vector of expected free energies, one per policy
    H_A = entropy(A)            # can calculate the entropy of the A matrix beforehand, since it'll be the same for all policies

    for policy_id, policy in enumerate(policies): # loop over policies - policy_id will be the linear index of the policy (0, 1, 2, ...) and `policy` will be a column vector where `policy[t,0]` indexes the action entailed by that policy at time `t`

        t_horizon = policy.shape[0] # temporal depth of the policy

        G_pi = 0.0 # initialize expected free energy for this policy

        for t in range(t_horizon): # loop over temporal depth of the policy

            action = policy[t,0] # action entailed by this particular policy, at time `t`

            # get the past predictive posterior - which is either your current posterior at the current time (not the policy time) or the predictive posterior entailed by this policy, one timstep ago (in policy time)
            if t == 0:
                qs_prev = qs_current
            else:
                qs_prev = qs_pi_t

            qs_pi_t = get_expected_states(B, qs_prev, action) # expected states, under the action entailed by the policy at this particular time
            qo_pi_t = get_expected_observations(A, qs_pi_t)   # expected observations, under the action entailed by the policy at this particular time

            kld = kl_divergence(qo_pi_t, C) # Kullback-Leibler divergence between expected observations and the prior preferences C

            G_pi_t = H_A.dot(qs_pi_t) + kld # predicted uncertainty + predicted divergence, for this policy & timepoint

            G_pi += G_pi_t # accumulate the expected free energy for each timepoint into the overall EFE for the policy

        G[policy_id] += G_pi

    return G

def compute_prob_actions(actions, policies, Q_pi):
    P_u = np.zeros(len(actions)) # initialize the vector of probabilities of each action

    for policy_id, policy in enumerate(policies):
        P_u[int(policy[0,0])] += Q_pi[policy_id] # get the marginal probability for the given action, entailed by this policy at the first timestep

    P_u = utils.norm_dist(P_u) # normalize the action probabilities

    return P_u

def active_inference_with_planning(A, B, C, D, n_actions, env, policy_len = 2, T = 5):

    """ Initialize prior, first observation, and policies """

    prior = D # initial prior should be the D vector

    obs = env.reset() # get the initial observation

    policies = construct_policies([n_states], [n_actions], policy_len = policy_len)

    for t in range(T):

        print(f'Time {t}: Agent observes itself in location: {obs}')

        # convert the observation into the agent's observational state space (in terms of 0 through 8)
        obs_idx = grid_locations.index(obs)

        # perform inference over hidden states
        qs_current = infer_states(obs_idx, A, prior)
        plot_beliefs(qs_current, title_str = f"Beliefs about location at time {t}", ax = next(axes))

        # calculate expected free energy of actions
        G = calculate_G_policies(A, B, C, qs_current, policies)

        # to get action posterior, we marginalize P(u|pi) with the probabilities of each policy Q(pi), given by \sigma(-G)
        Q_pi = softmax(-G)

        # compute the probability of each action
        P_u = compute_prob_actions(actions, policies, Q_pi)

        # sample action from probability distribution over actions
        chosen_action = utils.sample(P_u)

        # compute prior for next timestep of inference
        prior = B[:,:,chosen_action].dot(qs_current)

        # step the generative process and get new observation
        action_label = actions[chosen_action]
        obs = env.step(action_label)

    return qs_current


if __name__=='__main__':
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    with AxisIterator(n_rows=6,n_columns=6,figs=args.figs,title = 'Active inference from scratch') as axes:

        #  create a simple categorical distribution
        my_categorical = rng.random(size=3)
        my_categorical = utils.norm_dist(my_categorical)

        sampled_outcome = utils.sample(my_categorical)
        plot_beliefs(my_categorical, title_str = f'Random categorical distribution, sample={sampled_outcome}', ax = next(axes))

        # Now let’s move onto conditional categorical distributions or likelihoods,
        # i.e. how we represent the distribution of one discrete random variable X,
        # conditioned on the settings of another discrete random variable Y.
        p_x_given_y = rng.random((3, 4))
        p_x_given_y = utils.norm_dist(p_x_given_y)
        print(p_x_given_y[:,0])
        print(p_x_given_y[:,0].reshape(-1,1))
        print(f'Integral of P(X|Y=0): {p_x_given_y[:,0].sum()}')
        p_y = np.array([0.75, 0.25]) # this is already normalized - you don't need to `utils.norm_dist()` it!

        # the columns here are already normalized - you don't need to `utils.norm_dist()` it!
        p_x_given_y = np.array([[0.6, 0.5],
                                [0.15, 0.41],
                                [0.25, 0.09]])

        print(p_y.round(3).reshape(-1,1))
        print(p_x_given_y.round(3))
        E_x_wrt_y = p_x_given_y.dot(p_y)
        print(E_x_wrt_y)
        print(f'Integral: {E_x_wrt_y.sum().round(3)}')

        # A simple environment: Grid-world
        # Create  the grid locations in the form of a list of (Y, X) tuples
        grid_locations = list(product(range(3), repeat = 2))
        plot_grid(grid_locations,ax = next(axes))

        n_states = len(grid_locations)
        n_observations = len(grid_locations)

        # The A matrix or $A=P(o|s)
        # Create an umambiguous or 'noise-less' mapping between hidden states and observations
        A = np.eye(n_observations, n_states)
        plot_likelihood(A, title_str = '$A=P(o|s)$',ax = next(axes))

        #  Introduce ambiguity or uncertainty into the agent’s model of the world
        A_noisy = A.copy()
        A_noisy[0,0] = 1.0/3.0
        A_noisy[1,0] = 1.0/3.0
        A_noisy[3,0] = 1.0/3.0
        plot_likelihood(A_noisy, title_str = 'Blurred A matrix',ax = next(axes))

        my_A_noisy = A_noisy.copy()

        # locations 3 and 7 are the nearest neighbours to location 6
        my_A_noisy[3,6] = 1.0 / 3.0
        my_A_noisy[6,6] = 1.0 / 3.0
        my_A_noisy[7,6] = 1.0 / 3.0
        plot_likelihood(my_A_noisy, title_str = 'Noisy A matrix', ax = next(axes))

        B,actions = create_B_matrix()

        starting_location = (1,0)
        state_index = grid_locations.index(starting_location)
        starting_state = utils.onehot(state_index, n_states)
        plot_point_on_grid(starting_state, grid_locations, ax = next(axes))

        plot_beliefs(starting_state, 'Categorical distribution over the starting state', ax = next(axes))

        right_action_idx = actions.index('RIGHT')
        next_state = B[:,:, right_action_idx].dot(starting_state) # input the indices to the B matrix
        plot_point_on_grid(next_state, grid_locations, ax = next(axes))

        prev_state = next_state.copy()
        down_action_index = actions.index('DOWN')
        next_state = B[:,:,down_action_index].dot(prev_state)
        plot_point_on_grid(next_state, grid_locations,ax = next(axes))

        #  preferences over observations
        C = np.zeros(n_observations)
        desired_location = (2,2) # choose a desired location
        desired_location_index = grid_locations.index(desired_location) # get the linear index of the grid location, in terms of 0 through 8

        C[desired_location_index] = 1.0
        plot_beliefs(C, title_str = 'Preferences over observations', ax = next(axes))

        #  prior belief over hidden states at the first timestep
        D = utils.onehot(0, n_states)

        # demonstrate hwo belief about initial state can also be uncertain / spread among different possible initial states
        # alternative, where you have a degenerate/noisy prior belief
        # D = utils.norm_dist(np.ones(n_states))

        plot_beliefs(D, title_str = 'Prior beliefs over states', ax = next(axes))

        qs_past = utils.onehot(4, n_states) # agent believes they were at location 4 -- i.e. (1,1) one timestep ago

        last_action = 'UP' # the agent knew it moved 'UP' one timestep ago
        action_id = actions.index(last_action) # get the action index for moving 'UP'
        prior = B[:,:,action_id].dot(qs_past)

        observation_index = 1

        qs_new = infer_states(observation_index, A, prior)
        plot_beliefs(qs_new, title_str = 'Beliefs about hidden states', ax = next(axes))

        observation_index = 2 # this is like the agent is seeing itself in location (0, 2)
        qs_new = infer_states(observation_index, A, prior)
        plot_beliefs(qs_new, ax = next(axes))

        A_partially_ambiguous = softmax(A)
        noisy_prior = softmax(prior)
        plot_beliefs(noisy_prior, ax = next(axes))
        qs_new = infer_states(observation_index, A_partially_ambiguous, noisy_prior)
        plot_beliefs(qs_new, ax = next(axes))

        # Now let’s imagine we’re in some starting state, like (1,1)
        # N.B. This is the generative process we’re talking about – i.e. the true state of the world
        state_idx = grid_locations.index((1,1))
        state_vector = utils.onehot(state_idx, n_states)
        plot_point_on_grid(state_vector, grid_locations, ax = next(axes))

        # Make qs_current identical to the true starting state
        qs_current = state_vector.copy()
        plot_beliefs(qs_current, title_str ='Where do we believe we are?', ax = next(axes))

        #Create a preference to be in (1,2)

        desired_idx = grid_locations.index((1,2))

        C = utils.onehot(desired_idx, n_observations)

        plot_beliefs(C, title_str = 'Preferences', ax = next(axes))

        left_idx = actions.index('LEFT')
        right_idx = actions.index('RIGHT')

        print(f'Action index of moving left: {left_idx}')
        print(f'Action index of moving right: {right_idx}')

        ''' Compute the expected free energies for moving left vs. moving right '''
        G = np.zeros(2) # store the expected free energies for each action in here

        '''
        Compute G for MOVE LEFT here
        '''

        qs_u_left = get_expected_states(B, qs_current, left_idx)
        # alternative
        # qs_u_left = B[:,:,left_idx].dot(qs_current)

        H_A = entropy(A)
        qo_u_left = get_expected_observations(A, qs_u_left)
        # alternative
        # qo_u_left = A.dot(qs_u_left)

        predicted_uncertainty_left = H_A.dot(qs_u_left)
        predicted_divergence_left = kl_divergence(qo_u_left, C)
        G[0] = predicted_uncertainty_left + predicted_divergence_left

        '''
        Compute G for MOVE RIGHT here
        '''

        qs_u_right = get_expected_states(B, qs_current, right_idx)
        # alternative
        # qs_u_right = B[:,:,right_idx].dot(qs_current)

        H_A = entropy(A)
        qo_u_right = get_expected_observations(A, qs_u_right)
        # alternative
        # qo_u_right = A.dot(qs_u_right)

        predicted_uncertainty_right = H_A.dot(qs_u_right)
        predicted_divergence_right = kl_divergence(qo_u_right, C)
        G[1] = predicted_uncertainty_right + predicted_divergence_right

        # Now let's print the expected free energies for the two actions, that we just calculated
        print(f'Expected free energy of moving left: {G[0]}\n')
        print(f'Expected free energy of moving right: {G[1]}\n')

        Q_u = softmax(-G)

        print(f'Probability of moving left: {Q_u[0]}')
        print(f'Probability of moving right: {Q_u[1]}')

        env = GridWorldEnv()

        # To have everything in one place, let’s re-create the whole generative model
        A = np.eye(n_observations, n_states)

        B,actions = create_B_matrix()

        C = utils.onehot(grid_locations.index( (2, 2) ), n_observations) # make the agent prefer location (2,2) (lower right corner of grid world)

        D = utils.onehot(grid_locations.index( (1,2) ), n_states) # start the agent with the prior belief that it starts in location (1,2)

        # actions = ["UP", "DOWN", "LEFT", "RIGHT", "STAY"]

        env = GridWorldEnv(starting_state = (1,2))

        def run_active_inference_loop(A, B, C, D, actions, env, T = 5):
            """ Write a function that, when called, runs the entire active inference loop for a desired number of timesteps"""

            #Initialize the prior that will be passed in during inference to be the same as `D`
            prior = D.copy() # initial prior should be the D vector

            #Initialize the observation that will be passed in during inference - hint use env.reset()
            obs = env.reset() # initialize the `obs` variable to be the first observation you sample from the environment, before `step`-ing it.

            for t in range(T):

                print(f'Time {t}: Agent observes itself in location: {obs}')

                # convert the observation into the agent's observational state space (in terms of 0 through 8)
                obs_idx = grid_locations.index(obs)

                # perform inference over hidden states
                qs_current = infer_states(obs_idx, A, prior)

                plot_beliefs(qs_current, title_str = f"Beliefs about location at time {t}", ax = next(axes))

                # calculate expected free energy of actions
                G = calculate_G(A, B, C, qs_current, actions)

                # compute action posterior
                Q_u = softmax(-G)

                # sample action from probability distribution over actions
                chosen_action = utils.sample(Q_u)

                # compute prior for next timestep of inference
                prior = B[:,:,chosen_action].dot(qs_current)

                # update generative process
                action_label = actions[chosen_action]

                obs = env.step(action_label)

            return qs_current
        qs = run_active_inference_loop(A, B, C, D, actions, env, T = 5)

        policy_len = 4
        n_actions = len(actions)

        # we have to wrap `n_states` and `n_actions` in a list for reasons that will become clear in Part II
        all_policies = construct_policies([n_states], [n_actions], policy_len = policy_len)

        print(f'Total number of policies for {n_actions} possible actions and a planning horizon of {policy_len}: {len(all_policies)}')

        D = utils.onehot(grid_locations.index((0,0)), n_states) # let's have the agent believe it starts in location (0,0) (upper left corner)
        env = GridWorldEnv(starting_state = (0,0))
        qs_final = active_inference_with_planning(A, B, C, D, n_actions, env, policy_len = 3, T = 10)

    if args.show:
        show()

