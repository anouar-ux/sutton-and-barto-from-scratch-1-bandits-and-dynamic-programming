"""
Sutton and Barto from Scratch 1: Bandits and Dynamic Programming

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - create_bandit_testbed
def create_bandit_testbed(k, seed, mean=0.0, std=1.0):
    
    #Initialization
    rng=np.random.RandomState(seed)

    #Drawing
    values=np.array(rng.normal(loc=mean,scale=std,size=k))

    return values

# Step 2 - pull_arm
def pull_arm(true_values, action, rng):
    """Pull one arm and return reward = true value + unit-normal noise.

    Args:
        true_values (np.ndarray): Shape (k,) true mean reward of each arm.
        action (int): Index of the arm to pull.
        rng (np.random.Generator): Seeded random generator for the noise.

    Returns:
        float: Stochastic reward for this pull.
    """
    
    return true_values[action]+rng.normal(0,1)

# Step 3 - sample_average_update
def sample_average_update(q_values, action_counts, action, reward):
    
    q_val=q_values.copy()
    a_count=action_counts.copy()
    a_count[action]+=1
    n=a_count[action]
    val=q_val[action]

    q_val[action]+=(reward-val)/n
    
    return (q_val,a_count)

# Step 4 - epsilon_greedy_action
import numpy as np

def epsilon_greedy_action(q_values, epsilon, rng):
    # TODO: Choose an action epsilon-greedily from the current value estimates.
    random=rng.random()
    if random<epsilon:
        return rng.integers(0,q_values.shape)
    else:
        return int(np.argmax(q_values))

# Step 5 - run_bandit_episode
def run_bandit_episode(true_values, n_steps, epsilon, rng):
    """Run one bandit episode with epsilon-greedy selection and sample-average updates.

    Args:
        true_values (np.ndarray): Shape (k,) true mean reward of each arm.
        n_steps (int): Number of pulls in the episode.
        epsilon (float): Exploration probability for epsilon-greedy.
        rng (np.random.Generator): Seeded random generator.

    Returns:
        tuple: (rewards, actions) with shapes (n_steps,) and (n_steps,) of ints.
    """
    
    n=true_values.shape[0]
    q_table=np.zeros(n)
    a_count=np.zeros(n)
    reward_list=np.zeros(n_steps)
    action_list=np.zeros(n_steps)

    for k in range(n_steps):
        action=epsilon_greedy_action(q_table,epsilon,rng)
        action_list[k]=action
        reward=pull_arm(true_values,action,rng)
        reward_list[k]=reward
        q_table,a_count=sample_average_update(q_table,a_count,action,reward)

    return (reward_list,action_list)

# Step 6 - track_rewards_and_optimal_actions (not yet solved)
# TODO: implement

# Step 7 - average_bandit_curves (not yet solved)
# TODO: implement

# Step 8 - apply_random_walk_drift (not yet solved)
# TODO: implement

# Step 9 - constant_step_size_update (not yet solved)
# TODO: implement

# Step 10 - optimistic_initialization (not yet solved)
# TODO: implement

# Step 11 - ucb_action_select (not yet solved)
# TODO: implement

# Step 12 - gradient_bandit_update (not yet solved)
# TODO: implement

# Step 13 - bandit_parameter_study (not yet solved)
# TODO: implement

# Step 14 - build_gridworld_mdp (not yet solved)
# TODO: implement

# Step 15 - iterative_policy_evaluation (not yet solved)
# TODO: implement

# Step 16 - greedy_policy_improvement (not yet solved)
# TODO: implement

# Step 17 - policy_iteration (not yet solved)
# TODO: implement

# Step 18 - value_iteration (not yet solved)
# TODO: implement

# Step 19 - build_gambler_mdp (not yet solved)
# TODO: implement

# Step 20 - gambler_value_iteration (not yet solved)
# TODO: implement

# Step 21 - extract_optimal_stakes (not yet solved)
# TODO: implement

