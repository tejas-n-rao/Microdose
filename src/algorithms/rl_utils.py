# Shared tabular helpers and the MicroDose training adapter.


import random
from collections import defaultdict
import numpy as np
from src.simulation import microdose_env as env
from src.simulation.rl_environment import BenchmarkWorld, ACTIONS
from src.simulation.rewards import SAFETY_REWARD

N_STATES = env.GRID * env.GRID
N_ACTIONS = len(ACTIONS)



def state_index(x, y):
    # Referring to Q learning/packbot_env.py: flatten the state, using grid cells instead of wear.
    # This essentially replaces PackBot's (item_type, wear) index with microscope (x, y).
    return y * env.GRID + x



def init_Q():
    # Q learning/rl_utils.py: same table init with zeroes, with 2,500 grid states here.
    return np.zeros((N_STATES, N_ACTIONS))



def epsilon_greedy(Q, state, epsilon, rng):
    # retain the random-action versus argmax choice, same reference
    s = state_index(*state)
    if rng.random() < epsilon:
        return rng.randrange(N_ACTIONS)
    return int(np.argmax(Q[s]))



def greedy_action(Q, state):
    return int(np.argmax(Q[state_index(*state)]))



def linear_epsilon_schedule(episode, total_episodes, eps_start=1.0, eps_end=0.05):
    # exploration reaches its minimum after 80% of training.
    frac = min(1.0, episode / max(1, total_episodes * 0.8))
    return eps_start + frac * (eps_end - eps_start)



class MicroDoseControlEnv:


    def __init__(self, world, objective, rng, gamma, num_episodes):
        self.world, self.rng = world, rng
        if objective != 'safety':
            raise ValueError('Only the Safety objective is supported')
        self.weights = SAFETY_REWARD
        self.gamma, self.num_episodes = gamma, num_episodes


    def reset(self, episode=0):
        world = self.world

        # Project curriculum expands the random starts away from the tumour.
        radius = max(4, int(max(world.distance.values()) * min(1, 2*episode/self.num_episodes)))
        candidates = [p for p in world.cells if world.distance[p] <= radius]
        self.actual = world.start if episode % 5 == 0 else self.rng.choice(candidates)
        self.estimated = self.actual
        self.tick = 0
        self.done = False
        self.initial_terminal = world.terminal(self.actual, self.estimated, self.rng)
        return self.estimated


    def step(self, action):
        if self.done:
            raise RuntimeError('Reset before stepping a terminal episode')
        world = self.world
        state = self.estimated
        terminal = self.initial_terminal
        self.initial_terminal = None
        if terminal:
            # Random-start curriculum may start within the tumour radius.
            reward = sum(self.weights.components(terminal=terminal,
                         therapeutic_value=world.therapy(0)).values())
            self.done = True
            return state, reward + 3*world.distance[state], True, {'raw_reward': reward, 'terminal': terminal}
        self.actual, self.estimated, collision, drift, h, f, accidental = world.move(
            self.actual, ACTIONS[action], self.rng, self.tick)
        self.tick += 1
        terminal = ('delivery' if env.tumour_reached(self.actual, world.target) else 'off_target') if accidental else world.terminal(self.actual, self.estimated, self.rng)
        if terminal is None and self.tick == world.max_steps:
            terminal = 'timeout'
        reward = sum(self.weights.components(hazard=h, flow=f, collision=collision,
            movement=True, terminal=terminal, therapeutic_value=world.therapy(self.tick)).values())
        # Project shaping guides training. Reported returns keep the raw reward.
        potential_next = 0 if terminal else -3*world.distance[self.estimated]
        shaped_reward = reward + self.gamma*potential_next + 3*world.distance[state]
        self.done = terminal is not None
        return self.estimated, shaped_reward, self.done, {'raw_reward': reward, 'terminal': terminal}



def policy_snapshot(Q, world):
    # MicroDose replacement for the lab's PackBot-specific wear_threshold log.
    # follows rl_utils.py almost like a blackbox: wear_threshold logs become grid-start snapshots.
    s = state_index(*world.start)
    return {'start_action': greedy_action(Q, world.start), 'start_values': Q[s].tolist()}



# Q learning/rl_utils.py: unused value summary based on greedy_action, added here.
def _greedy_values_note(Q, positions):
    return {position: float(np.max(Q[state_index(*position)]))
            for position in positions}



def training_setup(world, objective, num_episodes, alpha, gamma, seed,
                   eps_start, eps_end, log_every, initial_Q=None, rng=None):
    if num_episodes < 1 or log_every < 1:
        raise ValueError('num_episodes and log_every must be positive')
    if not 0 < alpha <= 1 or not 0 <= gamma <= 1 or not 0 <= eps_end <= eps_start <= 1:
        raise ValueError('Invalid learning rate, discount or epsilon bounds')
    world = world if world is not None else BenchmarkWorld(0, 'deterministic')
    rng = rng if rng is not None else random.Random(seed)
    Q = init_Q() if initial_Q is None else np.array(initial_Q, dtype=float, copy=True)
    if Q.shape != (N_STATES, N_ACTIONS):
        raise ValueError('Q must have shape (N_STATES, N_ACTIONS)')
    return MicroDoseControlEnv(world, objective, rng, gamma, num_episodes), rng, Q



class TabularPolicy:


    # Compatibility interface for benchmark selection and existing checkpoints.
    def __init__(self, objective, seed=0):
        self.objective = objective
        if objective != 'safety':
            raise ValueError('Only the Safety objective is supported')
        self.weights = SAFETY_REWARD
        self.rng = random.Random(seed)
        self.q = defaultdict(lambda: np.zeros(N_ACTIONS))
        self.training_curve = []
        self.policy_log = []


    def select_action(self, observation):
        return ACTIONS[int(np.argmax(self.q.get(observation['estimated'], np.zeros(N_ACTIONS))))]


    def _train(self, trainer, world, episodes, **kwargs):
        Q = init_Q()

        for position, values in self.q.items():
            Q[state_index(*position)] = values
        Q, returns, log = trainer(num_episodes=episodes, world=world, objective=self.objective,
                                  initial_Q=Q, rng=self.rng, **kwargs)
        self.q = defaultdict(lambda: np.zeros(N_ACTIONS),
                             {p: Q[state_index(*p)].copy() for p in world.healthy})
        offset = len(self.training_curve)
        self.training_curve.extend(returns)
        self.policy_log.extend((episode+offset, snapshot) for episode, snapshot in log)
        return self


    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            algorithm = str(data['algorithm']) if 'algorithm' in data else 'q_learning'
            if algorithm != cls.algorithm:
                raise ValueError(f'Checkpoint uses {algorithm}; load with the corresponding policy class')
            policy = cls(str(data['objective']))
            policy.q.update({tuple(int(x) for x in state): values.copy()
                             for state, values in zip(data['states'], data['q'])})
            policy.training_curve = data['training_returns'].tolist()
        return policy


    def save(self, path):
        states = sorted(self.q)
        np.savez_compressed(path, states=np.asarray(states), q=np.asarray([self.q[s] for s in states]),
                            objective=self.objective, algorithm=self.algorithm, training_returns=self.training_curve)
