# Verify mission compatibility, dosage and controller admission.


import hashlib
import itertools
import json
from decimal import Decimal, ROUND_DOWN
from pathlib import Path
from typing import Literal

from src.verification.dosage_verifier import add_dosage_constraints
from src.verification.rejection_explanations import explain_rejection

from src.simulation.environment_sensor import is_very_high_alert

import z3
from pydantic import BaseModel, Field, ValidationError

RULES_PATH = Path(__file__).resolve().parents[2] / 'config' / 'mission_rules.json'
RULES = json.loads(RULES_PATH.read_text())
ASTAR = 'Speed · A*'
Q_SAFETY = 'Q-learning · Safety'
CONTROLLERS = (ASTAR, Q_SAFETY, 'SARSA · Safety')
UNSEEN_LAYOUTS = (5, 6, 7)



class StrictSpec(BaseModel):
    model_config = {'extra': 'forbid'}



# Z3Lab/verifier.py: typed request fields replace PackageSpec and AGVSpec.
class MissionAssignment(StrictSpec):
    target_id: Literal['A', 'B', 'C']
    carrier_id: Literal['X', 'Y', 'Z']
    dose: float = Field(default=50, gt=0, allow_inf_nan=False)
    preference: Literal['safety', 'speed']
    controller: str | None = None
    urgent: bool | None = None



class VerificationPayload(StrictSpec):
    missions: list[MissionAssignment] = Field(min_length=1, max_length=3)
    layout: int = Field(ge=0, le=7, strict=True)
    mode: Literal['deterministic', 'low', 'stochastic', 'very_high'] = 'deterministic'
    experimental: bool = False
    urgent_target: Literal['A', 'B', 'C'] | None = None
    urgency_evidence: str | None = None
    execution: Literal['simultaneous', 'sequential'] = 'simultaneous'



class MissionRuleViolation(ValueError):
    # A mission failed its SMT admission check.
    pass



def plan_fingerprint(raw_payload, sensor_observation=None):
    # Bind explicit compatibility consent to the complete normalized plan.
    payload = VerificationPayload.model_validate(raw_payload)
    identity = payload.model_dump(exclude={'urgency_evidence'})
    if sensor_observation:
        identity['sensor_observation'] = sensor_observation
    identity['urgent_target'] = payload.urgent_target or RULES['urgent_tumour']

    for row in identity['missions']:
        row.pop('urgent', None)
        if row['controller'] is None:
            row['controller'] = Q_SAFETY if is_very_high_alert(sensor_observation) or (payload.layout not in UNSEEN_LAYOUTS and row['preference'] == 'safety') else ASTAR
    if payload.execution == 'simultaneous':
        identity['missions'].sort(key=lambda row: row['carrier_id'])
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()



def compatibility_approval_options(raw_payload, sensor_observation=None):
    # Concrete mismatches for the UI to request consent. not authorizations.
    payload = VerificationPayload.model_validate(raw_payload)
    fingerprint = plan_fingerprint(raw_payload, sensor_observation)
    return [dict(carrier_id=m.carrier_id, target_id=m.target_id, dose=m.dose,
                 reagents=RULES['carrier_reagents'][m.carrier_id], tumour_type=RULES['tumour_types'][m.target_id],
                 plan_hash=fingerprint) for m in payload.missions
            if not set(RULES['carrier_reagents'][m.carrier_id]) & set(RULES['compatible_reagents'][RULES['tumour_types'][m.target_id]])]



def _approved(approvals, fingerprint, mission):
    return next((a for a in approvals or [] if a.get('plan_hash') == fingerprint
                 and a.get('carrier_id') == mission.carrier_id and a.get('target_id') == mission.target_id
                 and a.get('dose') == mission.dose and a.get('approved') is True
                 and bool(a.get('acknowledgement'))), None)



def verify_dispatch_smt(raw_payload, *, urgency_authorized=False, compatibility_approvals=None, parent_plan_hash=None,
                        sensor_observation=None, sensor_approval=None, planning_only=False):
    # Lab-compatible status/assignments/diagnostics/feedback response contract.
    # Z3Lab/verifier.py: validate the payload before creating indexed assignments.

    try:
        payload = VerificationPayload.model_validate(raw_payload)
    except ValidationError as exc:
        return {'status': 'SYNTACTIC_ERROR', 'diagnostics': {'validation': str(exc)},
                'feedback': 'Correct the mission payload before verification.'}
    fingerprint = plan_fingerprint(raw_payload, sensor_observation)
    approval_scope = parent_plan_hash or fingerprint
    sensor_alert = is_very_high_alert(sensor_observation)
    sensor_accepted = bool(sensor_approval and sensor_approval.get('accepted') is True
        and sensor_approval.get('plan_hash') == approval_scope
        and sensor_observation and sensor_approval.get('observation_id') == sensor_observation.get('observation_id'))
    # Named assertions are a project addition for clearer rejection reasons.
    solver = z3.Solver()
    carriers = list(RULES['carrier_reagents'])
    types = list(RULES['compatible_reagents'])
    assignments = [z3.Int(f'assign_{i}') for i in range(len(payload.missions))]
    controllers = [z3.Int(f'controller_{i}') for i in range(len(payload.missions))]


    def rule(expression, name):
        solver.assert_and_track(expression, name)

    if sensor_alert and not planning_only:
        rule(z3.BoolVal(sensor_accepted), 'sensor_alert_requires_user_acceptance')

    tumour_ids = list(RULES['tumour_types'])
    urgent_target = z3.Int('urgent_target')
    urgent_flags = {name: z3.Bool('urgent_' + name) for name in tumour_ids}
    default_urgent = RULES['urgent_tumour']
    requested_urgent = payload.urgent_target or default_urgent
    rule(urgent_target == tumour_ids.index(requested_urgent), 'requested_urgent_tumour')

    for name in tumour_ids:
        rule(urgent_flags[name] == (urgent_target == tumour_ids.index(name)), 'urgency_of_' + name)
    rule(z3.Sum([z3.If(flag, 1, 0) for flag in urgent_flags.values()]) == 1, 'exactly_one_urgent_tumour')
    # Authorization is a backend argument, not an LLM-writable payload flag.
    # ReAct derives it from a quoted user request. ordinary SMT calls cannot
    # override the experiment's configured default by setting a payload field.
    authorized = urgency_authorized and bool(payload.urgency_evidence and payload.urgency_evidence.strip())
    rule(z3.Or(urgent_target == tumour_ids.index(default_urgent), z3.BoolVal(authorized)),
         'urgency_override_requires_explicit_user_request')

    for i, mission in enumerate(payload.missions):
        assigned, selected = assignments[i], controllers[i]
        kind = z3.Int(f'tumour_type_{i}')
        rule(z3.And(assigned >= 0, assigned < len(carriers)), f'carrier_bounds_{i}')
        rule(assigned == carriers.index(mission.carrier_id), f'requested_carrier_{i}')
        if mission.urgent is not None:
            rule(urgent_flags[mission.target_id] == z3.BoolVal(mission.urgent), f'requested_urgency_{i}')
        type_name = RULES['tumour_types'][mission.target_id]
        rule(kind == types.index(type_name), f'tumour_type_{i}')
        matches = []

        for type_index, tumour_type in enumerate(types):
            for carrier_index, carrier in enumerate(carriers):
                if set(RULES['carrier_reagents'][carrier]) & set(RULES['compatible_reagents'][tumour_type]):
                    matches.append(z3.And(kind == type_index, assigned == carrier_index))
        consent = _approved(compatibility_approvals, approval_scope, mission)
        rule(z3.Or(z3.Or(*matches), z3.BoolVal(consent is not None)), f'reagent_matches_tumour_type_{i}')
        rule(z3.Implies(z3.BoolVal(payload.layout in UNSEEN_LAYOUTS and not sensor_alert), selected == 0),
             f'unseen_requires_astar_{i}')
        if sensor_alert:
            rule(selected == 1, f'sensor_very_high_requires_q_safety_{i}')
        elif payload.experimental:
            rule(z3.And(selected >= 0, selected < len(CONTROLLERS)), f'known_controller_{i}')
        else:
            rule(z3.Or(selected == 0, selected == 1), f'react_allows_only_astar_or_q_safety_{i}')
            if payload.layout not in UNSEEN_LAYOUTS:
                rule(selected == (1 if mission.preference == 'safety' else 0), f'seen_intent_controller_{i}')
        if mission.controller is not None:
            index = CONTROLLERS.index(mission.controller) if mission.controller in CONTROLLERS else -1
            rule(selected == index, f'requested_controller_{i}')

    for i, mission in enumerate(payload.missions):
        rule(z3.RealVal(str(mission.dose)) <= z3.RealVal(str(RULES['carrier_capacities'][mission.carrier_id])), f'carrier_capacity_{i}')
    if payload.execution == 'sequential':
        for carrier in carriers:
            aggregate = sum((Decimal(str(m.dose)) for m in payload.missions if m.carrier_id == carrier), Decimal(0))
            rule(z3.RealVal(str(aggregate)) <= z3.RealVal(str(RULES['carrier_capacities'][carrier])),
                 f'aggregate_carrier_capacity_{carrier}')
    add_dosage_constraints(solver, [(m.target_id, m.dose) for m in payload.missions], RULES['required_doses'])
    if payload.execution == 'simultaneous':
        rule(z3.Distinct(*assignments), 'one_mission_per_carrier')

    # Z3Lab/verifier.py: check SAT, then decode the assignment indices.
    status = solver.check()
    if status == z3.sat:
        model = solver.model()
        verified = []

        for i, mission in enumerate(payload.missions):
            carrier = carriers[model.eval(assignments[i]).as_long()]
            chosen = CONTROLLERS[model.eval(controllers[i]).as_long()]
            matching = bool(set(RULES['carrier_reagents'][carrier]) & set(RULES['compatible_reagents'][RULES['tumour_types'][mission.target_id]]))
            consent = _approved(compatibility_approvals, approval_scope, mission)
            verified.append(dict(status='SAT', **mission.model_dump(exclude={'controller', 'urgent'}),
                sequence_index=i, execution=payload.execution,
                sensor_observation=sensor_observation, sensor_approval=sensor_approval if sensor_accepted else None,
                sensor_forced=sensor_alert, policy_mode=sensor_observation['predicted_mode'] if sensor_observation else payload.mode,
                urgent=z3.is_true(model.eval(urgent_flags[mission.target_id])),
                urgent_target=requested_urgent, urgency_overridden=requested_urgent != default_urgent,
                urgency_evidence=payload.urgency_evidence if requested_urgent != default_urgent else None,
                required_dose=RULES['required_doses'][mission.target_id], max_capacity=RULES['carrier_capacities'][carrier], plan_hash=approval_scope,
                compatibility_match=matching, compatibility_overridden=not matching and consent is not None,
                compatibility_approval=consent if not matching else None,
                tumour_type=RULES['tumour_types'][mission.target_id],
                reagents=RULES['carrier_reagents'][carrier], controller=chosen,
                layout=payload.layout, split='unseen' if payload.layout in UNSEEN_LAYOUTS else 'seen',
                reason='Sensor predicts very high stochasticity: Q-learning Safety is mandatory.' if sensor_alert else 'Unseen layout: distance A* required by transfer evidence.'
                if payload.layout in UNSEEN_LAYOUTS else 'Tissue protection takes precedence over travel time.'
                if mission.preference == 'safety' else 'Travel time takes precedence.'))
        return {'status': 'REQUIRES_SENSOR_ACCEPTANCE' if sensor_alert and not sensor_accepted else 'SATISFIABLE', 'assignments': verified, 'plan_hash': fingerprint,
                'diagnostics': {'layout': payload.layout, 'execution': payload.execution,
                    'sensor_observation': sensor_observation, 'sensor_forced': sensor_alert, 'sensor_accepted': sensor_accepted,
                    'urgent_target': requested_urgent, 'default_urgent_target': default_urgent, 'carriers': [m['carrier_id'] for m in verified],
                    'requested_dose_by_target': {t: sum(m.dose for m in payload.missions if m.target_id == t) for t in RULES['required_doses']},
                    'required_dose_by_target': RULES['required_doses']},
                'feedback': 'Routing and payload constraints passed; accept Q-learning Safety or cancel before deployment.'
                if sensor_alert and not sensor_accepted else 'Dosage, compatibility/explicit consent, carrier allocation and controllers verified.'}
    conflicts = [str(c) for c in solver.unsat_core()] if status == z3.unsat else []
    totals = {t: sum(m.dose for m in payload.missions if m.target_id == t) for t in RULES['required_doses']}
    violations = explain_rejection(solver, payload, RULES, CONTROLLERS, UNSEEN_LAYOUTS, conflicts, sensor_observation) if status == z3.unsat else []
    if violations:
        feedback = 'SMT rejected the mission because ' + ' '.join(v['explanation'] for v in violations)
    else:
        feedback = 'Z3 could not verify the mission.'
    return {'status': 'UNSATISFIABLE' if status == z3.unsat else 'UNKNOWN', 'assignments': [],
            'diagnostics': {'unsat_core': conflicts, 'requested_dose_by_target': totals,
                            'required_dose_by_target': RULES['required_doses'],
                            'violations': violations,
                            'compatibility_conflicts': compatibility_approval_options(raw_payload, sensor_observation), 'requested_missions': [m.model_dump() for m in payload.missions]},
            'feedback': feedback}



def propose_alternative_plans(raw_payload, *, urgency_authorized=False, sensor_observation=None):
    """Enumerate solver-verified carrier reassignments, smallest change first.

    Preserve target, scenario, layout and intent. Excess allocations also receive
    dose-reduction proposals. dosage changes always require user selection. Never deploy a proposal
    without new user specifications. With three carriers the search is bounded.
    """

    try:
        payload = VerificationPayload.model_validate(raw_payload)
    except ValidationError:
        return []
    plans = []
    original = tuple(m.carrier_id for m in payload.missions)
    variants = [(payload.model_dump(), 0)]
    reduced = payload.model_dump()
    changed_doses = 0

    for target, limit in RULES['required_doses'].items():
        rows = [m for m in reduced['missions'] if m['target_id'] == target]
        total = sum((Decimal(str(m['dose'])) for m in rows), Decimal(0))
        if total > Decimal(str(limit)):
            for row in rows:
                row['dose'] = float((Decimal(str(row['dose'])) * Decimal(str(limit)) / total).quantize(Decimal('0.000001'), rounding=ROUND_DOWN))
                changed_doses += 1
    if changed_doses:
        variants.append((reduced, changed_doses))

    for base, dose_changes in variants:
        allocations = (itertools.product(RULES['carrier_reagents'], repeat=len(payload.missions))
                       if payload.execution == 'sequential'
                       else itertools.permutations(RULES['carrier_reagents'], len(payload.missions)))

        for allocation in allocations:
            candidate = json.loads(json.dumps(base))

            for mission, carrier in zip(candidate['missions'], allocation):
                mission['carrier_id'] = carrier
                mission['controller'] = None
            result = verify_dispatch_smt(candidate, urgency_authorized=urgency_authorized,
                                         sensor_observation=sensor_observation, planning_only=True)
            if result['status'] not in ('SATISFIABLE', 'REQUIRES_SENSOR_ACCEPTANCE'):
                continue
            changes = sum(a != b for a, b in zip(original, allocation))
            if not changes and not dose_changes and all(m.controller is None or m.controller == r['controller']
                                   for m, r in zip(payload.missions, result['assignments'])):
                continue
            plans.append({'payload': candidate, 'assignments': result['assignments'], 'changed_carriers': changes,
                          'changed_doses': dose_changes, 'status': result['status'], 'requires_user_choice': True})
    plans.sort(key=lambda p: (p['changed_carriers']+p['changed_doses'], p['changed_carriers'], tuple(m['carrier_id'] for m in p['assignments'])))
    return [dict(plan_id=f'plan_{i+1}', **plan) for i, plan in enumerate(plans)]



class SMTVerifier:


    # Compatibility facade for existing single-carrier callers.
    def verify(self, target_id, carrier_id, layout, preference, controller=None, experimental=False, dose=50,
               urgent_target=None, urgency_authorized=False, urgency_evidence=None, compatibility_approvals=None, parent_plan_hash=None, mode='deterministic', sensor_observation=None, sensor_approval=None):
        result = verify_dispatch_smt(dict(missions=[dict(target_id=target_id, carrier_id=carrier_id,
            preference=preference, controller=controller, dose=dose)], layout=layout, mode=mode, experimental=experimental,
            urgent_target=urgent_target, urgency_evidence=urgency_evidence), urgency_authorized=urgency_authorized,
            compatibility_approvals=compatibility_approvals, parent_plan_hash=parent_plan_hash, sensor_observation=sensor_observation, sensor_approval=sensor_approval)
        if result['status'] != 'SATISFIABLE':
            raise MissionRuleViolation(result['feedback'] + ' ' + str(result['diagnostics']))
        return result['assignments'][0]
