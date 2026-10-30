"""Simultaneous single/fleet carrier deployment with one shared clock.

Frozen per-target policies propose actions. Z3 admits joint realized movement,
then all positions commit from the same snapshot. No training occurs here.
"""
import random
from src.simulation import live_controller as live
from src.simulation import microdose_env as env
from src.simulation.rewards import SAFETY_REWARD
from src.simulation.safety_metrics import SAFETY_CALIBRATION
from src.verification.smt_verifier import ASTAR, RULES, verify_dispatch_smt, MissionRuleViolation
from src.verification.timestep_verifier import resolve_timestep


def _build_task_state(mission, layout, mode, policy, urgency, approvals, admission,
                      ledger, sequence_total, *, spawn=None, tick=0, rng_salt=0,
                      experimental=False):
    state = live.build_live_state(target_id=mission.target_id, mode=mode, seed=layout, dose=mission.dose,
        planner_name=mission.controller, policy=policy, carrier_id=mission.carrier_id,
        preference=mission.preference, mission_origin='experiment' if experimental else 'react', urgency_context=urgency,
        compatibility_context={'approvals': approvals, 'plan_hash': admission['plan_hash']},
        sensor_observation=mission.verification.get('sensor_observation'),
        sensor_approval=mission.verification.get('sensor_approval'))
    world = state['world']
    spawn = spawn if spawn is not None else env.seeded_robot_start(layout, mission.carrier_id)
    world.start = spawn
    state.update(start=spawn, actual=spawn, estimated=spawn, actual_trace=[spawn], estimate_trace=[spawn],
        dosage_ledger=ledger, moves=0, concurrency_waits=0,
        vehicle_type='speed route' if mission.controller == ASTAR else 'safety route',
        reservation_priority=2 if mission.controller == ASTAR else 1,
        rng=random.Random(layout*1009+rng_salt+17),
        sequence_index=mission.sequence_index, sequence_total=sequence_total,
        execution=mission.execution)
    if tick:
        state['tick'] = tick
    if state['policy'] is None:
        state['plan'] = world.planner.search(spawn, world.goal, cost_fn=env.unit_transition_cost())
    live.initialize_reference_route(state)
    return state


def build_fleet_state(missions, layout, mode, policies=None, *, experimental=False):
    policies = policies or {}
    execution = missions[0].execution if missions else 'simultaneous'
    payload = dict(missions=[dict(target_id=m.target_id, carrier_id=m.carrier_id,
        dose=m.dose, preference=m.preference, controller=m.controller) for m in missions], layout=layout, mode=mode, experimental=experimental)
    payload['execution'] = execution
    urgency = missions[0].verification
    if len({m.verification.get('urgent_target', RULES['urgent_tumour']) for m in missions}) != 1:
        raise MissionRuleViolation('All concurrent missions must share one urgent tumour designation.')
    payload.update(urgent_target=urgency.get('urgent_target', RULES['urgent_tumour']),
                   urgency_evidence=urgency.get('urgency_evidence'))
    approvals = [m.verification['compatibility_approval'] for m in missions if m.verification.get('compatibility_approval')]
    sensor_observation = urgency.get('sensor_observation')
    sensor_approval = urgency.get('sensor_approval')
    if any(m.verification.get('sensor_observation') != sensor_observation
           or m.verification.get('sensor_approval') != sensor_approval for m in missions):
        raise MissionRuleViolation('All tasks must share one trusted sensor observation and acceptance receipt.')
    admission = verify_dispatch_smt(payload, urgency_authorized=bool(urgency.get('urgency_overridden')),
                                    compatibility_approvals=approvals, sensor_observation=sensor_observation, sensor_approval=sensor_approval)
    if admission['status'] != 'SATISFIABLE':
        raise MissionRuleViolation(admission['feedback'])
    states, occupied = {}, set()
    ledger = {target: 0. for target in RULES['required_doses']}
    grouped = {}
    for mission in missions:
        grouped.setdefault(mission.carrier_id, []).append(mission)
    queues = {carrier: rows[1:] for carrier, rows in grouped.items()}
    totals = {carrier: len(rows) for carrier, rows in grouped.items()}
    for index, (carrier, rows) in enumerate(sorted(grouped.items())):
        mission = rows[0]
        spawn = env.seeded_robot_start(layout, carrier)
        if spawn in occupied:
            probe = live.build_live_state(target_id=mission.target_id, mode=mode, seed=layout, dose=mission.dose,
                planner_name=mission.controller, policy=policies.get((carrier, mission.sequence_index), policies.get(carrier)),
                carrier_id=carrier, preference=mission.preference,
                mission_origin='experiment' if experimental else 'react', urgency_context=urgency,
                compatibility_context={'approvals': approvals, 'plan_hash': admission['plan_hash']},
                sensor_observation=mission.verification.get('sensor_observation'),
                sensor_approval=mission.verification.get('sensor_approval'))
            world = probe['world']
            spawn = next(p for p in sorted(world.cells, key=lambda p: (env.manhattan(p, world.start), p)) if p not in occupied)
        occupied.add(spawn)
        policy = policies.get((carrier, mission.sequence_index), policies.get(carrier))
        states[carrier] = _build_task_state(
            mission, layout, mode, policy, urgency, approvals, admission, ledger, totals[carrier],
            spawn=spawn, rng_salt=index, experimental=experimental)
    return dict(carriers=states, seed=layout, mode=mode, tick=0, max_steps=live.MAX_LIVE_STEPS,
                done=all(s['done'] for s in states.values()) and not any(queues.values()), admission=admission,
                conflict_rng=random.Random(layout+7831), events=[], history=[], queues=queues,
                sequence_totals=totals, policies=policies, completed_tasks=[], experimental=experimental,
                urgency=urgency, approvals=approvals, dosage_ledger=ledger)


def _advance_sequences(fleet):
    # Activate the next ordered task only after the previous delivery.
    for carrier, state in list(fleet['carriers'].items()):
        if not state['done'] or not fleet['queues'].get(carrier):
            continue
        fleet['completed_tasks'].append(dict(
            robot=carrier, sequence_index=state['sequence_index'], target_id=state['target_id'],
            dose=state['dose'], controller=state['planner_name'], terminal=state['terminal'],
            safety_score=state['safety_score']))
        if state['terminal'] != 'delivery':
            fleet['queues'][carrier].clear()
            continue
        mission = fleet['queues'][carrier].pop(0)
        policy = fleet['policies'].get((carrier, mission.sequence_index), fleet['policies'].get(carrier))
        fleet['carriers'][carrier] = _build_task_state(
            mission, fleet['seed'], fleet['mode'], policy, fleet['urgency'], fleet['approvals'],
            fleet['admission'], fleet['dosage_ledger'], fleet['sequence_totals'][carrier],
            spawn=state['actual'], tick=fleet['tick'], rng_salt=mission.sequence_index + 31,
            experimental=fleet['experimental'])


def _proposal(state, eligible, tick):
    current = state['actual']
    if not eligible:
        return dict(current=current, destination=current, path=[], eligible=False)
    world = state['world']
    action = live.select_live_action(state)
    moved, estimated, collision, drift, hazard, flow, accidental = world.move(current, action, state['rng'], tick)
    intended = (current[0]+action[0], current[1]+action[1])
    if collision:
        intended = current
    return dict(current=current, destination=moved, path=[intended, moved], eligible=True,
                action=action, estimated=estimated, collision=collision, drift=drift,
                hazard=hazard, flow=flow, accidental=accidental)


def step_fleet_state(fleet):
    # One shared timestep. all robots receive one movement opportunity.
    # Completion belongs to the entire deployment, never the first robot.
    # Derive it from all assignments instead of relying on a stale UI flag.
    _advance_sequences(fleet)
    fleet['done'] = all(s['done'] for s in fleet['carriers'].values()) and not any(fleet['queues'].values())
    if any(s.get('needs_react') for s in fleet['carriers'].values()):
        fleet['needs_react'] = True
    if fleet['done'] or fleet.get('needs_react'):
        return
    tick = fleet['tick']
    states = fleet['carriers']
    for state in states.values():
        if state['done']:
            continue
        terminal = state['world'].terminal(state['actual'], state['estimated'], state['rng'])
        if terminal:
            live.finish(state, terminal, 'reach sensor')
        elif tick >= fleet['max_steps']:
            live.finish(state, 'timeout', 'shared timestep limit')
    _advance_sequences(fleet)
    states = fleet['carriers']
    if any(s.get('needs_react') for s in states.values()):
        fleet['needs_react'] = True
        return
    if all(s['done'] for s in states.values()) and not any(fleet['queues'].values()):
        fleet['done'] = True
        return
    # Every proposal reads the same pre-commit position snapshot.
    proposals = {k: _proposal(s, not s['done'], tick) for k, s in states.items()}
    resolution = resolve_timestep(proposals, {k: s['reservation_priority'] for k, s in states.items()}, fleet['conflict_rng'])
    fleet['events'].append(dict(timestep=tick+1, **resolution))
    for k, state in states.items():
        if state['done']:
            continue
        p, world = proposals[k], state['world']
        granted = resolution['granted'][k]
        requested_move = p['eligible'] and p['destination'] != p['current']
        waiting = requested_move and not granted
        if granted:
            actual, estimated = p['destination'], p['estimated']
            hazard, flow, drift = p['hazard'], p['flow'], p['drift']
        else:
            actual = p['current']
            estimated = env.microscope_estimate(actual, world.profile, state['rng'], False, world.blocked)
            hazard, flow = float(world.hazard[actual[1], actual[0]]), float(world.flows[tick][actual[1], actual[0]])
            drift = False
        # Re-evaluate release at the committed position, including waits.
        probability = 1 - (1 - env.INPUTS['reward']['healthy_cell_release_probability'])
        accidental = world.healthy[actual] >= env.INPUTS['reward']['healthy_cell_hit_threshold'] and state['rng'].random() < probability
        collision = bool(p.get('collision', False))
        state['actual'], state['estimated'], state['tick'] = actual, estimated, tick+1
        state['moves'] += int(granted)
        state['concurrency_waits'] += int(waiting)
        state['actual_trace'].append(actual)
        state['estimate_trace'].append(estimated)
        state['hazard_exposure'] += hazard
        state['observed_risk_exposure'] += world.healthy[estimated]
        state['collision_attempts'] += int(collision)
        state['drift_events'] += int(drift)
        costs = SAFETY_REWARD.components(movement=True, hazard=hazard, flow=flow, collision=collision)
        state['reward'] += sum(costs.values())
        state['safety_score'] = SAFETY_CALIBRATION.score(state['observed_risk_exposure'], state['collision_attempts'])
        state['flow'] = world.flows[tick+1].tolist()
        state['last_step'] = dict(command=p.get('action', (0, 0)), actual=actual, estimated=estimated,
            wall_contact=collision, drifted=drift, yielded=waiting, timestep=tick+1)
        state['history'].append(dict(step=tick+1, reward=state['reward'], safety_score=state['safety_score']))
        state['log'].insert(0, f'Timestep {tick+1}: ' + ('yielded to reservation' if waiting else 'moved' if granted else 'held position'))
        terminal = ('delivery' if env.tumour_reached(actual, world.target) else 'off_target') if accidental else world.terminal(actual, estimated, state['rng'])
        if terminal:
            live.finish(state, terminal, 'healthy-cell release' if accidental else 'reach sensor')
    if any(s.get('needs_react') for s in states.values()):
        fleet['needs_react'] = True
    fleet['tick'] = tick+1
    for state in states.values():
        if not state['done'] and fleet['tick'] >= fleet['max_steps']:
            live.finish(state, 'timeout', 'shared timestep limit')
    for state in states.values():
        live.refresh_live_plan(state)
    _advance_sequences(fleet)
    states = fleet['carriers']
    fleet['done'] = all(s['done'] for s in states.values()) and not any(fleet['queues'].values())
    fleet['history'].append(dict(timestep=fleet['tick'], reward=sum(s['reward'] for s in states.values()),
        delivered=sum(s['terminal'] == 'delivery' for s in states.values())))
