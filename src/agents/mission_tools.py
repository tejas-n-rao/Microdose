# Mission grounding, verification and clarification tools.


import json
import re
from src.simulation.environment_sensor import is_very_high_alert
from src.robot_identity import robot_label, ROBOT_CARRIERS
from src.agents.mission_models import MissionSpec, FleetMissionSpec
from src.verification.smt_verifier import verify_dispatch_smt, propose_alternative_plans, compatibility_approval_options

REQUIRED_QUESTIONS = {
    'target_id': 'Which tumour should receive the payload: A, B, or C?',
    'dose': 'How many drug units should be delivered?',
    'carrier_id': 'Which robot: R1 (reagent X), R2 (reagent Y), or R3 (both reagents)?',
    'layout': 'Which shared geometry ID: seen 0–4 or unseen 5–7?',
    'mode': 'Which shared scenario: deterministic, low, high, or very high stochasticity?',
    'preference': 'Should tissue protection take precedence over travel time, or should the carrier arrive as quickly as possible?',
}



# helper generated with AI for simultaneous deployment
def requested_concurrent_robots(instruction):
    # Recognize a named fleet, including short forms such as 'R1 ... and R3'.
    concurrent = re.search(r'\b(simultaneously|concurrently|together|fleet|both)\b|\band\s+(?:(?:robot|carrier)\s+)?R[123]\b', instruction, re.I)
    if not concurrent:
        return set()
    names = re.findall(r'\b(R[123])\b|\b(?:robot|carrier)\s+([XYZ])\b', instruction, re.I)
    return {ROBOT_CARRIERS.get((robot or legacy).upper(), (robot or legacy).upper())
            for robot, legacy in names}



def extract_explicit_single_mission(instruction):
    """Ground explicit single-mission facts before asking the user.

    ReAct still chooses tool actions. This extractor prevents an `ask_user`
    action from discarding values stated verbatim in the request. Later chat
    turns occur later in the combined instruction and therefore take priority.
    Multi-robot requests remain the LLM/tool schema's responsibility.
    """
    sequential = extract_explicit_sequential_missions(instruction)
    if sequential:
        return sequential
    patterns = {
        'layout': r'\blayout\s+([0-7])\b',
        'carrier_id': r'\b(?:robot|carrier)\s+(R[123]|[XYZ])\b',
        'dose': r'\b(\d+(?:\.\d+)?)\s*(?:drug\s+)?units?\b',
        'target_id': r'\b(?:to(?:\s+tumou?r)?|tumou?r)\s*([ABC])\b',
    }
    matches = {field: list(re.finditer(pattern, instruction, re.I))
               for field, pattern in patterns.items()}
    
    # Keep fleet rows with the LLM instead of collapsing them into one mission.
    if len(requested_concurrent_robots(instruction)) > 1:
        return {}
    result = {}

    for field, found in matches.items():
        if not found:
            continue
        match = found[-1]
        value = match.group(1)
        if field == 'layout':
            value = int(value)
        elif field == 'dose':
            value = float(value)
        elif field == 'carrier_id':
            value = ROBOT_CARRIERS.get(value.upper(), value.upper())
        else:
            value = value.upper()
        result[field] = value
        result.setdefault('evidence', {})[field] = match.group(0)
    replacements = list(re.finditer(
        r'\b(?:use|choose|switch\s+to|change\s+to)\s+(?:robot|carrier)\s+(R[123]|[XYZ])\s+instead\s+of\s+(?:robot|carrier)\s+(?:R[123]|[XYZ])\b',
        instruction, re.I))
    if replacements:
        replacement = replacements[-1]
        name = replacement.group(1).upper()
        result['carrier_id'] = ROBOT_CARRIERS.get(name, name)
        result.setdefault('evidence', {})['carrier_id'] = replacement.group(0)
    safety = list(re.finditer(r'\b(?:protect|avoid damaging|minimi[sz]e damage|healthy tissue)\b[^.\n]*', instruction, re.I))
    speed = list(re.finditer(r'\b(?:arriv(?:e|es|ing)|get there|deliver)\b[^.\n]*(?:quickly|as soon as possible|fast(?:est)?)\b', instruction, re.I))
    priorities = [(m.start(), 'safety', m.group(0)) for m in safety]
    priorities.extend((m.start(), 'speed', m.group(0)) for m in speed)
    if priorities:
        _, result['preference'], quote = max(priorities)
        result.setdefault('evidence', {})['preference'] = quote
    return result



def extract_explicit_sequential_missions(instruction):
    # Ground clear ``then`` clauses without collapsing them to the last task.
    if not re.search(r'\bthen\b|\bafter(?:wards?|\s+that)\b', instruction, re.I):
        return {}
    layout_matches = list(re.finditer(r'\blayout\s+([0-7])\b', instruction, re.I))
    clauses = re.split(r'\b(?:and\s+)?then\b|\bafter(?:wards?|\s+that)\b', instruction, flags=re.I)
    if len(clauses) < 2:
        return {}
    rows = []
    inherited_carrier = None

    for clause in clauses:
        dose = re.search(r'\b(\d+(?:\.\d+)?)\s*(?:drug\s+)?units?\b', clause, re.I)
        target = re.search(r'\b(?:to\s+)?(?:tumou?r\s+)([ABC])\b', clause, re.I)
        carrier = re.search(r'\b(?:via|using|robot|carrier)\s+(R[123]|[XYZ])\b', clause, re.I)
        if carrier:
            name = carrier.group(1).upper()
            inherited_carrier = ROBOT_CARRIERS.get(name, name)
        if not dose or not target or not inherited_carrier:
            return {}
        priority = extract_explicit_single_mission(clause)
        rows.append({
            'dose': float(dose.group(1)), 'target_id': target.group(1).upper(),
            'carrier_id': inherited_carrier, 'preference': priority.get('preference'),
            'evidence': {
                **({'preference': priority['evidence']['preference']} if priority.get('preference') else {}),
                'dose': dose.group(0), 'target_id': target.group(0),
                'carrier_id': carrier.group(0) if carrier else rows[-1]['evidence']['carrier_id'],
            },
        })
    if len(rows) < 2 or not layout_matches:
        return {}
    shared = re.search(r'[^.\n]*(?:for both tasks|for all tasks)[^.\n]*', instruction, re.I)
    if shared:
        priority = extract_explicit_single_mission(shared.group(0))

        for row in rows:
            if not row['preference'] and priority.get('preference'):
                row['preference'] = priority['preference']
                row['evidence']['preference'] = priority['evidence']['preference']
    layout = layout_matches[-1]
    return {
        'layout': int(layout.group(1)), 'execution': 'sequential', 'missions': rows,
        'evidence': {'layout': layout.group(0)},
    }



class SeenPayloads:


    # Z3Lab/agent.py: we use the sorted JSON as the lookup key.
    def __init__(self):
        self.payloads = set()


    def seen(self, payload):
        return json.dumps(payload, sort_keys=True) in self.payloads


    def add(self, payload):
        self.payloads.add(json.dumps(payload, sort_keys=True))



class MissionContext:


    def __init__(self, instruction, compatibility_approvals=None, scenario_mode=None, sensor_observation=None, sensor_approval=None):
        self.sensor_observation = sensor_observation
        self.sensor_approval = sensor_approval
        self.scenario_mode = scenario_mode
        self.instruction = instruction
        self.compatibility_approvals = compatibility_approvals or []
        self.draft = None
        self.verified = None # verifier flag because a changed draft must pass SMT again before it can finish
            # this wsa a bug as changed requests were being accepted. 
        self.last_result = None
        self.alternatives = []
        self.seen = SeenPayloads()
        self.urgency_authorized = False
        self.explicit = extract_explicit_single_mission(instruction)


    def set_mission(self, **args):
        self.verified = None  # fresh request, fresh check
        self.draft = None
        self.last_result = None
        self.alternatives = []
        rows = args.get('missions')
        if re.search(r'\b(simultaneously|concurrently|together)\b', self.instruction, re.I):
            execution = 'simultaneous'
        elif re.search(r'\bthen\b|\bafter(?:wards?|\s+that)\b', self.instruction, re.I):
            execution = 'sequential'
        else:
            execution = args.get('execution') or 'simultaneous'
        if rows is not None and any(args.get(field) is not None for field in ('target_id', 'dose', 'carrier_id', 'preference')):
            return {'status': 'SYNTACTIC_ERROR', 'feedback': 'Use either one top-level mission or the missions list; do not mix both.'}
        if rows is None:
            rows = [{key: args.get(key) for key in ('target_id', 'dose', 'carrier_id', 'preference', 'evidence')}]
            if self.explicit:
                explicit_evidence = self.explicit.get('evidence', {})
                if not isinstance(rows[0].get('evidence'), dict):
                    rows[0]['evidence'] = {}

                for field in ('target_id', 'dose', 'carrier_id', 'preference'):
                    if field in self.explicit:
                        rows[0][field] = self.explicit[field]
                        rows[0].setdefault('evidence', {})[field] = explicit_evidence[field]
                if 'layout' in self.explicit:
                    args['layout'] = self.explicit['layout']
                    if not isinstance(args.get('evidence'), dict):
                        args['evidence'] = {}
                    args['evidence']['layout'] = explicit_evidence['layout']
        questions = []
        requested = requested_concurrent_robots(self.instruction)
        extracted = {ROBOT_CARRIERS.get(row.get('carrier_id'), row.get('carrier_id')) for row in rows}
        missing = requested - extracted
        if missing:
            questions.append('The simultaneous request also names ' + ', '.join(robot_label(k) for k in sorted(missing))
                             + '. Provide their target, dose and priority; do not deploy a partial fleet.')
        evidence = args.get('evidence', {})
        urgent_target = args.get('urgent_target')
        self.urgency_authorized = False


        def supported(field, row, quotes):
            quote = quotes.get(field, '').strip()
            return row.get(field) is not None and quote and quote.casefold() in self.instruction.casefold()

        for field in (('layout',) if self.scenario_mode else ('layout', 'mode')):
            if not supported(field, args, evidence):
                questions.append(REQUIRED_QUESTIONS[field])

        for i, row in enumerate(rows):
            for field in ('target_id', 'dose', 'carrier_id', 'preference'):
                if field == 'preference' and row.get(field) is None and is_very_high_alert(self.sensor_observation):
                    row[field] = 'safety'
                    continue
                if not supported(field, row, row.get('evidence') or {}):
                    prefix = f'Mission {i+1}: ' if len(rows) > 1 else ''
                    questions.append(prefix + REQUIRED_QUESTIONS[field])
        if urgent_target is not None:
            if not supported('urgent_target', args, evidence):
                questions.append('Which tumour should be urgent? Provide an explicit urgency override in the instruction.')
            else:
                self.urgency_authorized = True
        self.draft = dict(missions=[{k: row.get(k) for k in ('target_id', 'dose', 'carrier_id', 'preference')} for row in rows],
                          layout=args.get('layout'), mode=self.scenario_mode or args.get('mode'), urgent_target=urgent_target,
                          urgency_evidence=evidence.get('urgent_target') if self.urgency_authorized else None,
                          execution=execution)
        if questions:
            return {'status': 'NEEDS_CLARIFICATION', 'questions': questions}
        return {'status': 'DRAFT_EXTRACTED', 'mission': self.draft, 'next_action': 'verify_dispatch_smt'}


    def ask_user(self, questions):
        # Ask only for specifications that are genuinely absent.
        if self.last_result and self.last_result.get('status') == 'UNSATISFIABLE':
            return {'status': 'NEEDS_CLARIFICATION', 'questions': questions}
        if self.explicit:
            grounded = self.set_mission(**self.explicit)
            if grounded['status'] == 'DRAFT_EXTRACTED':
                return {**grounded, 'status': 'SPECIFICATIONS_PRESENT',
                        'feedback': 'The request already states these fields verbatim. Continue with verify_dispatch_smt; do not ask the user again.'}
            return grounded
        return {'status': 'NEEDS_CLARIFICATION', 'questions': questions}


    def verify(self, *, recheck=False):
        if self.draft is None:
            return {'status': 'SYNTACTIC_ERROR', 'feedback': 'Call set_mission first.'}
        if self.seen.seen(self.draft) and not recheck:
            return {'status': 'REPEATED_PAYLOAD', 'feedback': 'This exact payload was already checked. Reflect on its diagnostics or ask the user; do not repeat it.'}
        self.seen.add(self.draft)
        result = verify_dispatch_smt(self.draft, urgency_authorized=self.urgency_authorized, compatibility_approvals=self.compatibility_approvals, sensor_observation=self.sensor_observation,
            sensor_approval=self.sensor_approval, planning_only=True)
        self.last_result = result
        if result['status'] == 'SATISFIABLE':
            self.verified = result
            return {**result, 'status': 'VERIFIED_MISSION', 'smt_status': 'SATISFIABLE'}
        return result


    def propose_alternatives(self):
        if not self.last_result or self.last_result['status'] != 'UNSATISFIABLE':
            return {'status': 'NO_CONFLICT', 'plans': [], 'feedback': 'Verify the requested plan before proposing repairs.'}
        self.alternatives = propose_alternative_plans(self.draft, urgency_authorized=self.urgency_authorized, sensor_observation=self.sensor_observation)
        return {'status': 'ALTERNATIVE_PLANS', 'diagnostics': self.last_result['diagnostics'],
                'feedback': self.last_result['feedback'], 'plans': self.alternatives,
                'requires_user_choice': True}


    def rejection_questions(self):
        # Turn solver diagnostics into deterministic user-facing repair prompts.
        if not self.last_result or self.last_result.get('status') != 'UNSATISFIABLE':
            return []
        violations = self.last_result.get('diagnostics', {}).get('violations', [])
        if violations:
            return [v['explanation'] + ' ' + v['repair'] for v in violations]
        return [self.last_result['feedback'] + ' Revise the conflicting specification and verify again.']


    def override_options(self):
        if not self.last_result or self.last_result['status'] != 'UNSATISFIABLE':
            return []
        diagnostic = self.last_result['diagnostics']
        if any(dose > diagnostic['required_dose_by_target'][target]
               for target, dose in diagnostic['requested_dose_by_target'].items()):
            return []  # Correct dosage before considering a mismatch override.
        return [option for option in compatibility_approval_options(self.draft, self.sensor_observation)
                if not any(a.get('plan_hash') == option['plan_hash'] and a.get('carrier_id') == option['carrier_id']
                           and a.get('approved') is True for a in self.compatibility_approvals)]


    def request_compatibility_override(self):
        options = self.override_options()
        if not options:
            return {'status': 'NO_COMPATIBILITY_OVERRIDE', 'feedback': 'Verify first. Excess dose requires a correction; it cannot be overridden.'}
        questions = [f"Robot {robot_label(o['carrier_id'])} carries {o['reagents']}, which does not match tumour {o['target_id']} ({o['tumour_type']}). Choose a compatible plan, or explicitly approve this mismatch at {o['dose']:g} units for this exact plan." for o in options]
        return {'status': 'NEEDS_CLARIFICATION', 'questions': questions, 'compatibility_options': options}


    def mission_spec(self, transcript):
        missions = [MissionSpec(target_id=row['target_id'], dose=row['dose'], carrier_id=row['carrier_id'],
            layout=self.draft['layout'], mode=self.draft['mode'], preference=row['preference'],
            controller=row['controller'], verification=row, transcript=transcript,
            sequence_index=row.get('sequence_index', index), execution=self.draft['execution'])
            for index, row in enumerate(self.verified['assignments'])]
        return missions[0] if len(missions) == 1 else FleetMissionSpec(
            missions, self.draft['layout'], self.draft['mode'], transcript, self.draft['execution'])


# Rejected shortcut. A draft alone says nothing about SMT admission.
# ReAct Lab/react_loop_lab.py: finish handling needs an extra SMT gate here.
# def old_finish(context, transcript):
#     if context.draft is not None:
#         return context.mission_spec(transcript)



def make_tool_registry(context):
    # ReAct Lab/sriroute_tools_db.py: map action names to bound tool functions.
    return {'set_mission': context.set_mission, 'verify_mission': context.verify,
            'verify_dispatch_smt': context.verify, 'propose_alternatives': context.propose_alternatives,
            'request_compatibility_override': context.request_compatibility_override,
            'ask_user': context.ask_user}
