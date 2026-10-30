# Q-learning control for microscope-estimated grid states.
# Source attribution in commments below

import numpy as np
from src.algorithms.rl_utils import (
    epsilon_greedy, linear_epsilon_schedule, state_index,
    training_setup, policy_snapshot, TabularPolicy,
)

# Keep benchmark imports available to existing callers.
from src.simulation.rl_environment import BenchmarkWorld, ACTIONS, evaluate



def qlearning_update(Q, s, a, r, s_next, alpha, gamma):
    # Return the updated scalar without changing Q. None marks a terminal state.
    # Reference was Q learning/qlearning_lab.py: This is the same structure for max-next-action update.
    if s_next is not None:
        val = Q[s, a] + alpha * (r + gamma * np.max(Q[s_next]) - Q[s, a])
    else:
        val = Q[s, a] + alpha * (r - Q[s, a])  # no next step to look ahead to
    return val



def train_qlearning(num_episodes=4000, alpha=0.15, gamma=0.99, seed=0,
                    eps_start=0.8, eps_end=0.05, log_every=50, *,
                    world=None, objective='safety', initial_Q=None, rng=None):
    """Mirrorring the lab's loop structure
        i.e., init S / choose A / step / update / S<-S'.

    Return (Q, episode_returns, policy_log), replacing wear thresholds with
    start-position policy shots
    """

    # qlearning_lab.py: reset, act, update, then advance S.
    env, rng, Q = training_setup(world, objective, num_episodes, alpha, gamma,
                                 seed, eps_start, eps_end, log_every, initial_Q, rng)
    episode_returns = []
    policy_log = []

    for i in range(num_episodes):
        s_init = env.reset(episode=i)
        s = state_index(*s_init)
        epsilon = linear_epsilon_schedule(i, num_episodes, eps_start, eps_end)
        done = False
        r_t = 0.0

        while not done:
            # Choose again each step, unlike the fixed action in the lab loop.
            action = epsilon_greedy(Q=Q, state=s_init, epsilon=epsilon, rng=rng)
            s_p, r, done, info = env.step(action=action)
            s_next = state_index(*s_p) if not done else None

            Q[s, action] = qlearning_update(Q=Q, s=s, a=action, r=r,
                                            s_next=s_next, alpha=alpha, gamma=gamma)
            s_init = s_p
            s = s_next
            r_t += info['raw_reward']

        episode_returns.append(r_t)

        # Grid snapshots replace the lab's wear-threshold logs.
        if (i + 1) % log_every == 0:
            policy_log.append((i, policy_snapshot(Q, env.world)))

    return Q, episode_returns, policy_log



# Scratch summary, kept for inspecting a cell without printing the whole table.
def _cell_values_note(Q, state):
    values = np.asarray(Q[state])
    return {'best_action': int(np.argmax(values)), 'spread': float(np.ptp(values))}


# Discarded attempt. argmax gives an action index, not its value.
# Compare Q learning/qlearning_lab.py, which uses the maximum value instead.
# def old_q_target(Q, s_next, reward, gamma):
#     return reward + gamma * np.argmax(Q[s_next])



class QLearningPolicy(TabularPolicy):
    algorithm = 'q_learning'


    def train(self, world, episodes=2000, **kwargs):
        return self._train(train_qlearning, world, episodes, **kwargs)


if __name__ == '__main__':
    Q, returns, snapshots = train_qlearning()
    print('Final start-position policy:', snapshots[-1][1])
    print('Mean return, last 200 episodes:', np.mean(returns[-200:]))
