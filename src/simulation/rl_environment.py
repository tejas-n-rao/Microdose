# MicroDose dynamics and evaluation shared by lab-structured control algorithms.
import random
import time
from collections import defaultdict, deque
import numpy as np
from src.simulation import microdose_env as env
from src.simulation.rewards import SAFETY_REWARD, therapy_components
from src.simulation.safety_metrics import SAFETY_CALIBRATION
from src.algorithms import AStarPlanner

ACTIONS = ((1, 0), (-1, 0), (0, 1), (0, -1))



class BenchmarkWorld:


    def __init__(self, layout, mode, target='B', max_steps=160, dose=50, carrier_id=None):
        self.layout, self.mode, self.max_steps, self.dose = layout, mode, max_steps, dose
        self.profile = env.ENVIRONMENT_PROFILES[mode]
        self.blocked = env.configured_blocked(layout)
        self.start = env.seeded_robot_start(layout, carrier_id)
        self.target = env.seeded_tumours(layout)[target]
        self.goal = self.target['true']
        h, f, _ = env.create_fields(layout)
        self.hazard = np.asarray(h)
        self.flows = [np.asarray(f)]
        yy, xx = np.indices(self.hazard.shape)

        for tick in range(1, max_steps + 1):
            previous = self.flows[-1]
            if self.profile.flow_source_scale:
                previous = np.clip(previous * self.profile.flow_decay +
                    np.abs(np.sin((tick + xx * 2 + yy) / 17)) * .004 * self.profile.flow_source_scale, 0, 1)
            self.flows.append(previous)
        blobs = env.seeded_healthy_tissue(layout)
        self.healthy = {p: env.healthy_tissue_risk_from_blobs(p, blobs)
                        for p in env.vessel_layout(layout).traversable}
        self.cells = sorted(self.healthy)
        # Unweighted map distance is a shaping potential, not an action policy.
        self.distance = {p: 0 for p in self.cells if env.tumour_reached(p, self.target)}
        frontier = deque(self.distance)

        while frontier:
            current = frontier.popleft()

            for adjacent in env.neighbors(current, self.blocked):
                if adjacent not in self.distance:
                    self.distance[adjacent] = self.distance[current] + 1
                    frontier.append(adjacent)
        self.cells = [p for p in self.cells if p in self.distance]
        # A noisy microscope can report a disconnected traversable pocket.
        unreachable_distance = max(self.distance.values()) + 1

        for position in self.healthy:
            self.distance.setdefault(position, unreachable_distance)
        self.planner = AStarPlanner(env.make_grid(h, f, self.blocked))
        self.plan_actions = {}


    def astar_action(self, estimated):
        # Distance-only feedback actions. live reference recovery is handled separately.
        key = estimated
        if key not in self.plan_actions:
            costs = env.unit_transition_cost()
            result = self.planner.search(estimated, self.goal, cost_fn=costs)

            for a, b in zip(result.path, result.path[1:]):
                self.plan_actions[a] = (b[0]-a[0], b[1]-a[1])
            self.plan_actions.setdefault(key, (0, 0))
        return self.plan_actions[key]


    def move(self, actual, action, rng, tick):
        intended = actual[0]+action[0], actual[1]+action[1]
        collision = not env.is_traversable(intended, self.blocked)
        if collision:
            intended = actual
        moved, drift = env.apply_slip(intended, self.profile, rng, self.blocked)
        estimated = env.microscope_estimate(moved, self.profile, rng, drift, self.blocked)
        h = float(self.hazard[moved[1], moved[0]])
        f = float(self.flows[tick][moved[1], moved[0]])
        accidental = (self.healthy[moved] >= env.INPUTS['reward']['healthy_cell_hit_threshold']
                      and rng.random() < env.INPUTS['reward']['healthy_cell_release_probability'])
        return moved, estimated, collision, drift, h, f, accidental


    def terminal(self, actual, estimated, rng):
        if env.release_sensor_thinks_reached(actual, estimated, self.target, rng):
            return 'delivery' if env.tumour_reached(actual, self.target) else 'off_target'
        return None


    def therapy(self, tick):
        return sum(therapy_components(self.dose, tick, env.INPUTS['reward']).values()) + env.near_tumour_release_reward(self.goal, self.target)



def evaluate(world, algorithm, seed, policy=None, calibration=SAFETY_CALIBRATION):
    rng = random.Random(seed)
    actual = estimated = world.start
    components = defaultdict(float)
    weights = SAFETY_REWARD
    metrics = dict(algorithm=algorithm, mode=world.mode, layout=world.layout, seed=seed,
                   steps=0, hazard_exposure=0., healthy_exposure=0., peak_hazard=0.,
                   high_risk_steps=0, collision_attempts=0, drift_events=0, flow_exposure=0., observed_risk_exposure=0., covered_steps=0)
    terminal = None
    started = time.perf_counter()

    for tick in range(world.max_steps):
        terminal = world.terminal(actual, estimated, rng)
        if terminal:
            break
        metrics['covered_steps'] += int(policy is None or estimated in policy.q)
        action = policy.select_action({'estimated': estimated}) if policy else world.astar_action(estimated)
        actual, estimated, collision, drift, h, f, accidental = world.move(actual, action, rng, tick)
        metrics['steps'] += 1
        metrics['hazard_exposure'] += h
        metrics['healthy_exposure'] += world.healthy[actual]
        metrics['observed_risk_exposure'] += world.healthy[estimated]
        metrics['flow_exposure'] += f
        metrics['peak_hazard'] = max(metrics['peak_hazard'], h)
        metrics['high_risk_steps'] += int(h >= .18)
        metrics['collision_attempts'] += int(collision)
        metrics['drift_events'] += int(drift)

        for key, value in weights.components(hazard=h, flow=f, collision=collision, movement=True).items():
            components[key] += value
        if accidental:
            terminal = 'delivery' if env.tumour_reached(actual, world.target) else 'off_target'
            break
    if terminal is None:
        terminal = world.terminal(actual, estimated, rng) or 'timeout'

    for key, value in weights.components(terminal=terminal, therapeutic_value=world.therapy(metrics['steps'])).items():
        components[key] += value
    metrics['safety_return'] = sum(components.values())
    metrics.update({'safety_'+key: value for key, value in components.items()})
    metrics.update(success=int(terminal == 'delivery'), off_target=int(terminal == 'off_target'),
                   timeout=int(terminal == 'timeout'), final_distance=env.manhattan(actual, world.goal),
                   runtime_ms=1000*(time.perf_counter()-started))
    metrics['safety_score'] = calibration.score(metrics['observed_risk_exposure'], metrics['collision_attempts'])
    metrics['policy_coverage'] = metrics['covered_steps']/max(1,metrics['steps'])
    return metrics
