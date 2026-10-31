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
    
    return float(true_values[action]+rng.normal(0,1))

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
        return int(rng.integers(0, len(q_values)))
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
    
    n=len(true_values)
    q_table=np.zeros(n)
    a_count=np.zeros(n)
    reward_list=np.zeros(n_steps)
    action_list=np.zeros(n_steps,dtype=int)

    for k in range(n_steps):
        action=epsilon_greedy_action(q_table,epsilon,rng)
        action_list[k]=action
        reward=pull_arm(true_values,action,rng)
        reward_list[k]=reward
        q_table,a_count=sample_average_update(q_table,a_count,action,reward)

    return (reward_list,action_list)

# Step 6 - track_rewards_and_optimal_actions
def track_rewards_and_optimal_actions(true_values, n_steps, epsilon, rng):
    """Run one episode tracking rewards and optimal-arm choices.

    Args:
        true_values (np.ndarray): Shape (k,) true mean reward of each arm.
        n_steps (int): Number of pulls in the episode.
        epsilon (float): Exploration probability for epsilon-greedy.
        rng (np.random.Generator): Seeded random generator.

    Returns:
        tuple: (rewards, optimal_flags) each shape (n_steps,).
            optimal_flags entries are 0.0 or 1.0 floats.
    """
    # TODO: return per-step rewards and 0/1 optimal-arm flags
    optimal_action=np.argmax(true_values)
    reward,actions=run_bandit_episode(true_values,n_steps,epsilon,rng)
    flag=(actions==optimal_action).astype(float)
    return (reward,flag)

# Step 7 - average_bandit_curves
def average_bandit_curves(k, n_runs, n_steps, epsilon, seed):
    
    
    rewards = np.zeros((n_runs,n_steps))
    flags = np.zeros((n_runs,n_steps))

    for l in range(n_runs):
        true_values=create_bandit_testbed(k,seed+l)
        rng = np.random.default_rng(seed + l)
        reward,flag= track_rewards_and_optimal_actions(true_values, n_steps, epsilon, rng)
        rewards[l]=reward
        flags[l]=flag

    r_mean = np.mean(rewards, axis=0)
    f_mean = np.mean(flags, axis=0)

    return (r_mean, f_mean)

# Step 8 - apply_random_walk_drift
def apply_random_walk_drift(true_values, drift_std, rng):
    

    noise=rng.normal(loc=0,scale=drift_std,size=true_values.shape)

    return true_values+noise

# Step 9 - constant_step_size_update
def constant_step_size_update(q_values, action, reward, alpha):
    

    val=q_values[action]
    val+= alpha*(reward-val)
    q_values[action]=val

    return q_values

# Step 10 - optimistic_initialization
def optimistic_initialization(k, initial_value):
    
    return initial_value*np.ones(k)

# Step 11 - ucb_action_select
def ucb_action_select(q_values, action_counts, timestep, c):
    """Select an action by upper-confidence-bound scores.

    Args:
        q_values (np.ndarray): Action-value estimates, shape (k,).
        action_counts (np.ndarray): Visit counts per action, shape (k,).
        timestep (int): Current time step t (>= 1).
        c (float): Exploration constant.

    Returns:
        int: Index of the selected action.
    """
    n=len(q_values)

    values =  np.zeros(n)

    for k in range(n):
        N=action_counts[k]
        if N==0:
            return k
        val= q_values[k] + c*(np.log(timestep)/N)**0.5
        values[k]=val

    return int(np.argmax(values))

# Step 12 - gradient_bandit_update
def gradient_bandit_update(preferences, action, reward, average_reward, alpha):
    
    def softmax(values):
        val=values-np.max(values)
        tot = np.sum(np.exp(val))
        return np.exp(val)/tot

    pi = softmax(preferences)
    adv=reward - average_reward
    preferences -= alpha*adv*pi
    preferences[action] += alpha*adv

    return preferences

# Step 13 - bandit_parameter_study
import numpy as np

def bandit_parameter_study(n_runs,n_steps,seed, settings,   k=10 ):
    """
    Run a multi-method bandit comparison over specified settings.
    """
    rng = np.random.default_rng(seed)
    results = {}

    for setting in settings:
        method = setting['method']
        param = setting['param']
        nonstationary = setting.get('nonstationary', False)

        label = f"{method}({param})"
        if nonstationary:
            label += ",ns"

        final_step_rewards = np.zeros(n_runs)

        # --- INITIALISATION DE L'ENVIRONNEMENT (Stationnaire) ---
        # Créé une seule fois par setting
        if not nonstationary:
            base_true_values = create_bandit_testbed(k, seed=seed)

        for run in range(n_runs):
            # --- RESET DE L'ENVIRONNEMENT ---
            if nonstationary:
                true_values = np.zeros(k)
            else:
                # On utilise une copie pour s'assurer que le tableau de base 
                # n'est jamais altéré par accident lors des manipulations
                true_values = np.copy(base_true_values)

            # 2. Initialize agent parameters
            if method == 'optimistic':
                q_values = optimistic_initialization(k, param)
            else:
                q_values = np.zeros(k)
                
            action_counts = np.zeros(k, dtype=int)
            preferences = np.zeros(k)  # Used only for gradient method
            average_reward = 0.0       # Baseline for gradient method

            last_reward = 0.0

            # 3. Run the episode
            for t in range(1, n_steps + 1):
                
                # --- ACTION SELECTION ---
                if method == 'epsilon_greedy':
                    action = epsilon_greedy_action(q_values, param, rng)
                elif method == 'constant_step':
                    action = epsilon_greedy_action(q_values, 0.1, rng)
                elif method == 'optimistic':
                    action = epsilon_greedy_action(q_values, 0.0, rng)
                elif method == 'ucb':
                    action = ucb_action_select(q_values, action_counts, t, param)
                elif method == 'gradient':
                    exp_prefs = np.exp(preferences - np.max(preferences))
                    probs = exp_prefs / np.sum(exp_prefs)
                    action = rng.choice(k, p=probs)
                else:
                    raise ValueError(f"Unknown method: {method}")

                # --- EXECUTE ACTION ---
                reward = pull_arm(true_values, action, rng)
                if t == n_steps:
                    last_reward = reward

                # --- AGENT UPDATE ---
                action_counts[action] += 1
                
                if method in ['epsilon_greedy', 'ucb']:
                    sample_average_update(q_values, action_counts, action, reward)
                elif method == 'constant_step':
                    constant_step_size_update(q_values, action, reward, param)
                elif method == 'optimistic':
                    constant_step_size_update(q_values, action, reward, 0.1)
                elif method == 'gradient':
                    average_reward += (reward - average_reward) / t
                    gradient_bandit_update(preferences, action, reward, average_reward, param)

                # --- ENVIRONMENT UPDATE (Nonstationary) ---
                if nonstationary:
                    apply_random_walk_drift(true_values, drift_std=0.01, rng=rng)
                    
            final_step_rewards[run] = last_reward
            
        results[label] = np.mean(final_step_rewards)
        
    return results
    rng = np.random.default_rng(seed)
    results = {}

    for setting in settings:
        method = setting['method']
        param = setting['param']
        nonstationary = setting.get('nonstationary', False)

        # Generate setting label
        label = f"{method}({param})"
        if nonstationary:
            label += ",ns"

        final_step_rewards = np.zeros(n_runs)

        for run in range(n_runs):
            # 1. Initialize environment
            if nonstationary:
                true_values = np.zeros(k)
            else:
                run_seed = int(rng.integers(100000000))
                true_values = create_bandit_testbed(k, seed=run_seed)

            # 2. Initialize agent parameters
            if method == 'optimistic':
                q_values = optimistic_initialization(k, param)
            else:
                q_values = np.zeros(k)
                
            action_counts = np.zeros(k, dtype=int)
            preferences = np.zeros(k)  # Used only for gradient method
            average_reward = 0.0       # Baseline for gradient method

            # Track reward at the final step
            last_reward = 0.0

            # 3. Run the episode
            for t in range(1, n_steps + 1):
                
                # --- ACTION SELECTION ---
                if method == 'epsilon_greedy':
                    action = epsilon_greedy_action(q_values, param, rng)
                elif method == 'constant_step':
                    action = epsilon_greedy_action(q_values, 0.1, rng)
                elif method == 'optimistic':
                    action = epsilon_greedy_action(q_values, 0.0, rng)
                elif method == 'ucb':
                    action = ucb_action_select(q_values, action_counts, t, param)
                elif method == 'gradient':
                    # Softmax selection (with numerical stability shift)
                    exp_prefs = np.exp(preferences - np.max(preferences))
                    probs = exp_prefs / np.sum(exp_prefs)
                    action = rng.choice(k, p=probs)
                else:
                    raise ValueError(f"Unknown method: {method}")

                # --- EXECUTE ACTION ---
                reward = pull_arm(true_values, action, rng)
                if t == n_steps:
                    last_reward = reward

                # --- AGENT UPDATE ---
                action_counts[action] += 1
                
                if method in ['epsilon_greedy', 'ucb']:
                    sample_average_update(q_values, action_counts, action, reward)
                elif method == 'constant_step':
                    constant_step_size_update(q_values, action, reward, param)
                elif method == 'optimistic':
                    constant_step_size_update(q_values, action, reward, 0.1)
                elif method == 'gradient':
                    # Incremental average reward including current reward
                    average_reward += (reward - average_reward) / t
                    gradient_bandit_update(preferences, action, reward, average_reward, param)

                # --- ENVIRONMENT UPDATE (Nonstationary) ---
                if nonstationary:
                    apply_random_walk_drift(true_values, drift_std=0.01, rng=rng)
                    
            final_step_rewards[run] = last_reward
            
        # Store mean of the final step reward across all runs
        results[label] = np.mean(final_step_rewards)
        
    return results

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

