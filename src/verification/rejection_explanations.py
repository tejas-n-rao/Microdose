# Explain conflicting constraints using concrete mission choices.


from collections import Counter
import z3
from src.robot_identity import robot_label



# deried from Z3Lab/verifier.py
def explain_rejection(solver, payload, rules, controllers, unseen_layouts, core, sensor_observation=None):
    carriers = list(rules['carrier_reagents'])
    types = list(rules['compatible_reagents'])
    tumour_ids = list(rules['tumour_types'])
    urgent = payload.urgent_target or rules['urgent_tumour']
    facts = [z3.Int('urgent_target') == tumour_ids.index(urgent)]
    facts.extend(z3.Bool('urgent_'+target) == (target == urgent) for target in tumour_ids)

    for i, mission in enumerate(payload.missions):
        if mission.controller:
            effective = mission.controller
        elif sensor_observation and sensor_observation.get('predicted_mode') == 'very_high':
            effective = controllers[1]
        else:
            effective = controllers[1] if payload.layout not in unseen_layouts and mission.preference == 'safety' else controllers[0]
        selected = controllers.index(effective) if effective in controllers else -1
        facts.extend([z3.Int(f'assign_{i}') == carriers.index(mission.carrier_id),
                      z3.Int(f'controller_{i}') == selected,
                      z3.Int(f'tumour_type_{i}') == types.index(rules['tumour_types'][mission.target_id])])

    # similar to snippets in Z3 labs verifier.py
    # helps explain concrete conflicts. Per-rule violations however were present in the lab, and the Z3 docs had a similar example
    violations = []

    for assertion in solver.assertions():
        # assert_and_track stores Implies(tracking_literal, constraint).
        rule_id, expression = str(assertion.arg(0)), assertion.arg(1)
        probe = z3.Solver()
        probe.add(*facts, expression)
        if probe.check() == z3.unsat:
            detail = _describe(rule_id, payload, rules, controllers)
            violations.append(dict(rule_id=rule_id, solver_status='UNSATISFIABLE',
                in_unsat_core=rule_id in core, **detail))
    return violations



def _describe(rule_id, payload, rules, controllers):
    detail = dict(override_allowed=False, mission_indices=[])
    if rule_id.startswith('dosage_budget_'):
        target = rule_id.removeprefix('dosage_budget_')
        rows = [(i, m) for i, m in enumerate(payload.missions) if m.target_id == target]
        total = sum(m.dose for _, m in rows)
        cap = rules['required_doses'][target]
        robots = ', '.join(robot_label(m.carrier_id) for _, m in rows)
        return dict(**detail, explanation=f'Tumour {target} is allocated {total:g} units across {robots}, exceeding its shared {cap:g}-unit dose limit by {total-cap:g} units.',
            repair=f'Reduce the combined allocation to at most {cap:g} units and verify again. This limit cannot be overridden.',
            evidence=dict(target=target, requested=total, limit=cap, excess=total-cap))
    if rule_id == 'one_mission_per_carrier':
        duplicates = [k for k, count in Counter(m.carrier_id for m in payload.missions).items() if count > 1]
        robots = ', '.join(robot_label(k) for k in duplicates)
        return dict(**detail, explanation=f'{robots} is assigned more than one simultaneous mission. Each robot can carry out only one mission per deployment.',
            repair='Assign these missions to distinct compatible robots and verify the entire fleet again.', evidence=dict(duplicate_robots=duplicates))
    if rule_id.startswith('aggregate_carrier_capacity_'):
        carrier = rule_id.removeprefix('aggregate_carrier_capacity_')
        total = sum(m.dose for m in payload.missions if m.carrier_id == carrier)
        cap = rules['carrier_capacities'][carrier]
        return dict(**detail,
            explanation=f'Robot {robot_label(carrier)} is allocated {total:g} units across its sequence, exceeding its {cap:g}-unit aggregate capacity by {total-cap:g} units.',
            repair=f'Reduce the combined sequence payload to at most {cap:g} units. Capacity cannot be overridden.',
            evidence=dict(robot=robot_label(carrier), requested=total, limit=cap))
    if rule_id == 'urgency_override_requires_explicit_user_request':
        return dict(**detail, explanation=f'The request changes the urgent tumour from {rules["urgent_tumour"]} to {payload.urgent_target} without an authorized explicit urgency instruction.',
            repair=f'Explicitly request which tumour should be urgent, or retain the default {rules["urgent_tumour"]}, then verify again.', evidence=dict(default=rules['urgent_tumour'], requested=payload.urgent_target))
    if rule_id == 'sensor_alert_requires_user_acceptance':
        return dict(**detail, explanation='Sensors predict very high stochasticity. A* is unavailable and Q-learning Safety requires explicit user acceptance.',
            repair='Accept the proposed Q-learning Safety plan or cancel deployment. This restriction cannot be bypassed by a speed request.', evidence={})
    i = int(rule_id.rsplit('_', 1)[-1])
    mission = payload.missions[i]
    robot = robot_label(mission.carrier_id)
    detail['mission_indices'] = [i]
    if rule_id.startswith('reagent_matches_tumour_type_'):
        kind = rules['tumour_types'][mission.target_id]
        carried = rules['carrier_reagents'][mission.carrier_id]
        required = rules['compatible_reagents'][kind]
        compatible = [robot_label(k) for k, reagents in rules['carrier_reagents'].items() if set(reagents) & set(required)]
        detail['override_allowed'] = True
        return dict(**detail, explanation=f'Robot {robot} carries reagent {" + ".join(carried)}. Tumour {mission.target_id} is {kind.replace("_", " ")} and requires reagent {" or ".join(required)}; the carried reagent does not match.',
            repair=f'Choose compatible robot {" or ".join(compatible)}, change the target, or explicitly request a plan-scoped mismatch override. Correct any excess dose before permission is offered.',
            evidence=dict(robot=robot, target=mission.target_id, tumour_type=kind, carried=carried, required=required, compatible_robots=compatible))
    if rule_id.startswith('carrier_capacity_'):
        cap = rules['carrier_capacities'][mission.carrier_id]
        return dict(**detail, explanation=f'Robot {robot} is allocated {mission.dose:g} units, exceeding its {cap:g}-unit payload capacity by {mission.dose-cap:g} units.',
            repair=f'Reduce this robot’s payload to at most {cap:g} units while respecting shared tumour limits. Capacity cannot be overridden.', evidence=dict(robot=robot, requested=mission.dose, limit=cap))
    if rule_id.startswith('sensor_very_high_requires_q_safety_'):
        return dict(**detail, explanation=f'Sensors predict very high stochasticity. Robot {robot} requests {mission.controller}; only Q-learning Safety is admitted.',
            repair='Use Q-learning Safety and explicitly accept the plan, or cancel deployment.', evidence=dict(robot=robot, required=controllers[1]))
    if rule_id.startswith('unseen_requires_astar_'):
        return dict(**detail, explanation=f'Robot {robot} requests {mission.controller} on UNSEEN layout {payload.layout}. SMT requires distance A* on unseen layouts.',
            repair='Use distance A* on this layout, or choose a seen layout for an RL experiment.', evidence=dict(robot=robot, layout=payload.layout, requested=mission.controller, required=controllers[0]))
    if rule_id.startswith(('react_allows_only_astar_or_q_safety_', 'known_controller_')):
        allowed = controllers if payload.experimental else controllers[:2]
        return dict(**detail, explanation=f'Robot {robot} requests controller {mission.controller}, which is not admitted for this deployment.',
            repair='Choose an admitted controller: '+', '.join(allowed)+'.', evidence=dict(robot=robot, requested=mission.controller, allowed=list(allowed)))
    if rule_id.startswith('seen_intent_controller_'):
        required = controllers[1] if mission.preference == 'safety' else controllers[0]
        return dict(**detail, explanation=f'Robot {robot} requests {mission.controller}, but its {mission.preference} priority on a seen layout requires {required}.',
            repair='Confirm the intended priority and use its admitted controller; structured experiments can compare other saved policies.', evidence=dict(robot=robot, priority=mission.preference, requested=mission.controller, required=required))
    if rule_id.startswith('requested_urgency_'):
        urgent = payload.urgent_target or rules['urgent_tumour']
        return dict(**detail, explanation=f'Robot {robot} marks tumour {mission.target_id} as urgent={mission.urgent}, conflicting with the single urgent designation {urgent}.',
            repair='Align the mission urgency with the shared designation, or explicitly request a new urgent tumour.', evidence=dict(robot=robot, target=mission.target_id, requested=mission.urgent, urgent_target=urgent))
    return dict(**detail, explanation=f'The concrete request violates SMT constraint {rule_id}.', repair='Revise the conflicting specification and verify again.', evidence={})
