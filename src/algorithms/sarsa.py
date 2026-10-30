# SARSA control with sampled next actions and grid states.

# Same structyure as as the q_learning file, but with this implementation of sarsa.


import numpy as np
from src.algorithms.rl_utils import (
    epsilon_greedy, linear_epsilon_schedule, state_index,
    training_setup, policy_snapshot, TabularPolicy,
)



def sarsa_update(Q, s, a, r, s_next, a_next, alpha, gamma):
    # Return the scalar update. Terminal transitions have no next state or action.
    # Reference: Q learning/sarsa_lab.py: bootstrap from the action actually sampled.
    if s_next is None:
        val = Q[s, a] + alpha * (r - Q[s, a])
    else:
        val = Q[s, a] + alpha * (r + gamma * Q[s_next, a_next] - Q[s, a])
    return val



def train_sarsa(num_episodes=4000, alpha=0.15, gamma=0.99, seed=0,
                eps_start=0.8, eps_end=0.05, log_every=50, *,
                world=None, objective='safety', initial_Q=None, rng=None):
    
    # carry both S and its sampled action into the loop.
    env, rng, Q = training_setup(world, objective, num_episodes, alpha, gamma,
                                 seed, eps_start, eps_end, log_every, initial_Q, rng)
    episode_returns = []
    policy_log = []

    for i in range(num_episodes):
        r_t = 0.0
        s_init = env.reset(episode=i)
        s = state_index(*s_init)
        epsilon = linear_epsilon_schedule(i, num_episodes, eps_start, eps_end)
        action = epsilon_greedy(Q=Q, state=s_init, epsilon=epsilon, rng=rng)
        done = False

        while not done:
            s_p, r, done, info = env.step(action=action)
            if not done:
                s_next = state_index(*s_p)
                action_next = epsilon_greedy(Q=Q, state=s_p, epsilon=epsilon, rng=rng)
            else:
                s_next = None
                action_next = None


            Q[s, action] = sarsa_update(Q=Q, s=s, a=action, r=r, s_next=s_next,
                                        a_next=action_next, alpha=alpha, gamma=gamma)
            s_init = s_p
            # Keep A' forthe next move
            s, action = s_next, action_next  # keep that pick for the next move
            r_t += info['raw_reward']

        episode_returns.append(r_t)
        if (i + 1) % log_every == 0:
            policy_log.append((i, policy_snapshot(Q, env.world)))

    return Q, episode_returns, policy_log


# Old idea, wrong for SARSA because it ignores the sampled next action.
# Contrast with Q learning/sarsa_lab.py, which uses a_next in the target.
# def old_sarsa_target(Q, s_next, reward, gamma):
#     return reward + gamma * np.max(Q[s_next])


# Q learning/sarsa_lab.py: the supplied loop uses fixed eps_start.
# Unused draft, before applying the shared decay schedule.
# def _fixed_exploration_action(Q, state, rng, eps_start=0.8):
#     return epsilon_greedy(Q, state, eps_start, rng)



class SARSAPolicy(TabularPolicy):
    algorithm = 'sarsa'


    def train(self, world, episodes=2000, **kwargs):
        return self._train(train_sarsa, world, episodes, **kwargs)


if __name__ == '__main__':
    Q, returns, snapshots = train_sarsa()
    print('Final start-position policy:', snapshots[-1][1])
    print('Mean return, last 200 episodes:', np.mean(returns[-200:]))
