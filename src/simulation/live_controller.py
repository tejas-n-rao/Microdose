# Live deployment of path planners and learned policies through shared dynamics.
import random
from src.algorithms import AStarResult
from src.simulation import microdose_env as env
from src.simulation.rl_environment import BenchmarkWorld
from src.simulation.rewards import SAFETY_REWARD

POLICIES = {
    'Q-learning · Safety': ('q_learning', 'safety'),
    'SARSA · Safety': ('sarsa', 'safety'),
}
PLANNERS = ('Speed · A*', *POLICIES)
MAX_LIVE_STEPS = 160



def build_live_state(target_id, mode, seed, dose, planner_name, policy=None,
                     carrier_id=None, preference='speed', mission_origin='experiment', urgency_context=None, compatibility_context=None, sensor_observation=None, sensor_approval=None):
    from src.verification.smt_verifier import SMTVerifier
    urgency_context = urgency_context or {}
    compatibility_context = compatibility_context or {}
    verification = SMTVerifier().verify(target_id, carrier_id, seed, preference,
        controller=planner_name, experimental=mission_origin != 'react', dose=dose,
        urgent_target=urgency_context.get('urgent_target'),
        urgency_authorized=mission_origin == 'react' and bool(urgency_context.get('urgency_overridden')),
        urgency_evidence=urgency_context.get('urgency_evidence'), mode=mode,
        compatibility_approvals=compatibility_context.get('approvals'),
        parent_plan_hash=compatibility_context.get('plan_hash'),
        sensor_observation=sensor_observation, sensor_approval=sensor_approval)
    if planner_name not in PLANNERS:
        raise ValueError('Unknown controller')
    if planner_name in POLICIES:
        learner, objective = POLICIES[planner_name]
        if policy is None or policy.algorithm != learner or policy.objective != objective:
            raise ValueError('Load a matching saved policy before starting this mission')
    world = BenchmarkWorld(seed, mode, target_id, MAX_LIVE_STEPS, dose, carrier_id=carrier_id)
    hazard, flow, variance = env.create_fields(seed)
    layout = env.vessel_layout(seed)
    planner = world.planner
    if policy is not None:
        plan = AStarResult([], 0, 0., 0., 'manhattan')
    else:
        cost = env.unit_transition_cost()
        plan = planner.search(world.start, world.goal, cost_fn=cost)
    state = dict(target_id=target_id, mode=mode, seed=seed, dose=dose, planner_name=planner_name,
        carrier_id=carrier_id, preference=preference, mission_origin=mission_origin,
        mission_verification=verification,
        policy=policy, world=world, planner=planner, plan=plan, heuristic='manhattan',
        rng=random.Random(seed), hazard=hazard, flow=flow, variance=variance,
        grid=world.planner.grid, blocked=world.blocked, boundary=set(layout.boundary),
        deadspace=set(layout.deadspace), center_chaos=env.center_chaos_lookup(seed),
        start=world.start, tumours=env.seeded_tumours(seed), actual=world.start, estimated=world.start,
        actual_trace=[world.start], estimate_trace=[world.start], plan_cursor=0,
        reward=0., hazard_exposure=0., observed_risk_exposure=0.,
        collision_attempts=0, drift_events=0, safety_score=100., tick=0, max_steps=MAX_LIVE_STEPS,
        delivered=False, done=False, terminal=None, release_actual_success=False,
        release_event=None, last_step=None, reagent_used=0., dosage_ledger={'A': 0., 'B': 0., 'C': 0.},
        allocated_dose=dose, needs_react=False,
        admitted_identity=(target_id, carrier_id, seed, mode, planner_name, preference, dose),
        compatibility_context=compatibility_context,
        history=[], log=[f'{planner_name} ready for tumour {target_id}.'])

    initialize_reference_route(state)
    return state



def initialize_reference_route(state):
    # Freeze the launch-time route. Learned policies retain their own actions.
    state['reference_path'] = tuple(state['plan'].path) if state['policy'] is None else ()
    state['plan_cursor'] = 0
    state['recovery_path'] = []
    state['navigation_mode'] = 'reference route' if state['reference_path'] else 'learned policy'



def refresh_live_plan(state):
    # Keep the reference fixed. recover using only the perceived position.
    if state['policy'] is not None or state['done']:
        return
    if 'reference_path' not in state:
        state['plan'] = state['planner'].search(state['start'], state['world'].goal,
                                               cost_fn=env.unit_transition_cost())
        initialize_reference_route(state)
    reference = state['reference_path']
    if not reference:
        state['navigation_mode'] = 'no reference route'
        return
    cursor = state['plan_cursor']
    estimated = state['estimated']
    previous_mode = state['navigation_mode']
    on_reference = next((i for i in range(cursor, len(reference)) if reference[i] == estimated), None)
    if on_reference is not None:
        state['plan_cursor'] = on_reference
        state['recovery_path'] = []
        state['navigation_mode'] = 'reference route'
        if previous_mode == 'rejoining reference':
            state['log'].insert(0, f'Reference rejoined at perceived cell {estimated}.')
        return
    waypoint = reference[min(cursor+1, len(reference)-1)]
    recovery = state['planner'].search(estimated, waypoint, cost_fn=env.unit_transition_cost())
    state['recovery_path'] = recovery.path
    state['navigation_mode'] = 'rejoining reference'
    if previous_mode != 'rejoining reference':
        state['log'].insert(0, f'Perceived wander at {estimated}; returning to reference waypoint {waypoint}.')



def select_live_action(state):
    if state['policy'] is not None:
        return state['policy'].select_action({'estimated': state['estimated']})
    refresh_live_plan(state)
    reference = state['reference_path']
    if not reference:
        return (0, 0)
    if state['navigation_mode'] == 'rejoining reference':
        recovery = state['recovery_path']
        if len(recovery) < 2:
            return (0, 0)
        a, b = recovery[:2]
    else:
        a = state['estimated']
        b = reference[min(state['plan_cursor']+1, len(reference)-1)]
    return b[0]-a[0], b[1]-a[1]



def finish(state, terminal, reason):
    if state['done']:
        return
    world = state['world']
    if terminal in ('delivery', 'off_target'):
        from src.verification.dosage_verifier import verify_dosage_smt
        from src.verification.smt_verifier import SMTVerifier, MissionRuleViolation
        result = verify_dosage_smt(dict(trajectories=[dict(carrier_id=state['carrier_id'],
            target_id=state['target_id'], allocated_dose=state['allocated_dose'],
            path=state['actual_trace'], releases=[dict(step=len(state['actual_trace'])-1, amount=state['dose'])])],
            previously_used=state['dosage_ledger']))
        identity = tuple(state[k] for k in ('target_id', 'carrier_id', 'seed', 'mode', 'planner_name', 'preference', 'dose'))

        try:
            if identity != state['admitted_identity']:
                raise MissionRuleViolation('Mission changed after admission. Revise and verify through ReAct before release.')
            if result['status'] != 'SATISFIABLE':
                raise MissionRuleViolation(result['feedback'])
            v = state['mission_verification']
            context = state['compatibility_context']
            SMTVerifier().verify(state['target_id'], state['carrier_id'], state['seed'], state['preference'],
                controller=state['planner_name'], experimental=state['mission_origin'] != 'react', dose=state['dose'],
                mode=state['mode'], urgent_target=v['urgent_target'], urgency_authorized=v['urgency_overridden'],
                urgency_evidence=v.get('urgency_evidence'), compatibility_approvals=context.get('approvals'),
                parent_plan_hash=context.get('plan_hash'),
                sensor_observation=v.get('sensor_observation'), sensor_approval=v.get('sensor_approval'))
        except MissionRuleViolation as exc:
            state.update(done=True, terminal='rejected', delivered=False, needs_react=True,
                         release_rejection={**result, 'status': 'UNSATISFIABLE', 'feedback': str(exc)})
            state['log'].insert(0, 'Release rejected: ' + str(exc))
            return
        state['dosage_ledger'][state['target_id']] += state['dose']
        state['reagent_used'] += state['dose']
    state['terminal'] = terminal
    state['done'] = True
    state['delivered'] = terminal in ('delivery', 'off_target')
    state['release_actual_success'] = terminal == 'delivery'
    state['reward'] += sum(SAFETY_REWARD.components(
        terminal=terminal, therapeutic_value=world.therapy(state['tick'])).values())
    if state['delivered']:
        state['release_event'] = dict(step=state['tick'], position=state['actual'],
            microscope_estimate=state['estimated'], target=state['target_id'],
            target_position=world.goal, reason=reason, actual_tumour_region=terminal == 'delivery', dose=state['dose'])
    refresh_live_plan(state)
    if state['history']:
        state['history'][-1]['reward'] = state['reward']
    elapsed = f"{state['tick']} shared timesteps, {state['moves']} moves" if 'moves' in state else f"{state['tick']} moves"
    state['log'].insert(0, f'Mission ended: {terminal}, after {elapsed}.')
