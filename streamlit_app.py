from __future__ import annotations

import os
import sys
import time
import uuid
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault('MPLCONFIGDIR', str(PROJECT_ROOT / '.cache' / 'matplotlib'))

import matplotlib.pyplot as plt

from src.agents.microdose_react import (
    FleetMissionSpec, MissionClarificationRequired, MissionParseError,
    has_openai_api_key, parse_mission_instruction,
    sample_natural_language_mode,
)
from src.agents.mission_models import MissionSpec
from src.simulation.environment_sensor import observe_environment
from src.policy_store import load_deployed_policy
from src.storage import (import_checkpoints, model_revision, start_deployment, sync_deployment,
                         cancel_deployment, record_event)
from src.algorithms.deployment import POLICY_DIR
from src.robot_identity import robot_label
from src.simulation.fleet_controller import build_fleet_state, step_fleet_state
from src.simulation.live_controller import POLICIES
from src.ui.fleet_view import render_fleet
from src.ui.grader_guide import render_grader_guide
from src.ui.live_views import initial_robot_states, plot_microscope_view, plot_world_preview
from src.ui.snapshot_browser import snapshot_state
from src.verification.smt_verifier import RULES, verify_dispatch_smt


@st.cache_resource(show_spinner=False)
def deployed_policy(controller, target, mode, layout, revision):
    return load_deployed_policy(controller, target, mode, layout)


def chat_defaults():
    return {
        'react_chat': [],
        'react_submit_requested': False,
        'mission_alternatives': [],
        'compatibility_options': [],
        'compatibility_approvals': [],
        'mission_smt_rejection': {},
        'nl_missions': None,
        'nl_verified': False,
        'nl_sampled_mode': None,
        'last_mission_transcript': [],
        'acknowledge_compatibility': False,
        'nl_plan_requested': False,
        'deployment_experimental': False,
        'deployment_source': 'react',
        'structured_error': '',
        'nl_sensor_observation': None,
        'sensor_approval': None,
        'sensor_prompt': {},
    }


def reset_react_chat():
    cancel_deployment(st.session_state.get('fleet_state'))
    for key, value in chat_defaults().items():
        st.session_state[key] = value
    for key in ('verified_instruction', 'fleet_state'):
        st.session_state.pop(key, None)


def react_instruction():
    turns = [message['content'].strip() for message in st.session_state.react_chat
             if message.get('role') == 'user' and message.get('content', '').strip()]
    if not turns:
        return ''
    return turns[0] + ''.join(
        '\nAdditional user specifications (these supersede earlier conflicting values): ' + turn
        for turn in turns[1:])


def approve_compatibility_override():
    if not st.session_state.get('acknowledge_compatibility'):
        return
    for option in st.session_state.compatibility_options:
        st.session_state.compatibility_approvals.append({
            **option, 'approved': True,
            'acknowledgement': 'User explicitly approved this listed reagent/type mismatch via the UI.',
        })
    st.session_state.compatibility_options = []
    st.session_state.acknowledge_compatibility = False
    st.session_state.nl_verified = False
    st.session_state.react_chat.append({
        'role': 'user',
        'content': 'I explicitly approve the listed reagent/type mismatch for this exact plan.',
    })
    st.session_state.react_submit_requested = True


def apply_alternative_plan(plan_id):
    proposal = next((plan for plan in st.session_state.mission_alternatives
                     if plan['plan_id'] == plan_id), None)
    if proposal is None:
        return
    option_number = next(index for index, plan in enumerate(
        st.session_state.mission_alternatives, start=1) if plan['plan_id'] == plan_id)
    rows = [
        f"Robot {robot_label(mission['carrier_id'])}: deliver {mission['dose']:g} units "
        f"to tumour {mission['target_id']}; "
        + ('protect healthy tissue even if the trip takes longer'
           if mission['preference'] == 'safety' else 'arrive as quickly as possible')
        for mission in proposal['assignments']
    ]
    if proposal['payload'].get('execution') == 'sequential':
        instruction = (f"On layout {proposal['payload']['layout']}, perform these tasks in order: "
                       + '; then '.join(rows))
    else:
        instruction = (f"On layout {proposal['payload']['layout']}, deploy these robots simultaneously: "
                       + '; '.join(rows))
    urgent = proposal['payload'].get('urgent_target')
    if urgent and urgent != RULES['urgent_tumour']:
        instruction += f". Treat tumour {urgent} urgently instead of {RULES['urgent_tumour']}."
    st.session_state.react_chat.append({
        'role': 'user', 'content': f"I choose Option {option_number}: {instruction}",
    })
    st.session_state.compatibility_approvals = []
    st.session_state.compatibility_options = []
    st.session_state.mission_alternatives = []
    st.session_state.mission_smt_rejection = {}
    st.session_state.nl_verified = False
    st.session_state.react_submit_requested = True


def rejection_message(error):
    if error.sensor_prompt:
        return '**Sensor alert: very high stochasticity predicted.**\n\nA* will not be used. No robot has been deployed. Choose **Accept Q-learning Safety** or **Cancel deployment** below.'
    if not error.smt_rejection:
        return '\n\n'.join([
            'I need more information before I can verify this mission.',
            *('- ' + question for question in error.questions),
        ])
    violations = error.smt_rejection.get('diagnostics', {}).get('violations', [])
    sections = ['**Request rejected by SMT. No robot was deployed.**']
    if violations:
        sections.append('**Why it was rejected**\n' + '\n'.join(
            '- ' + violation['explanation'] for violation in violations))
        sections.append('**What you can change**\n' + '\n'.join(
            '- ' + violation['repair'] for violation in violations))
    if error.alternatives:
        count = len(error.alternatives)
        sections.append(f"Choose one of the {count} verified alternative{'s' if count != 1 else ''} below, "
                        'or send a revised instruction.')
    return '\n\n'.join(sections)


def process_pending_mission():
    if not st.session_state.pop('react_submit_requested', False):
        return
    instruction = react_instruction()
    if st.session_state.nl_sampled_mode is None:
        st.session_state.nl_sampled_mode = sample_natural_language_mode()
    if st.session_state.nl_sensor_observation is None:
        st.session_state.nl_sensor_observation = observe_environment(st.session_state.nl_sampled_mode)
    st.session_state.nl_verified = False
    st.session_state.sensor_prompt = {}
    try:
        mission = parse_mission_instruction(
            instruction,
            compatibility_approvals=st.session_state.compatibility_approvals,
            scenario_mode=st.session_state.nl_sampled_mode,
            sensor_observation=st.session_state.nl_sensor_observation,
            sensor_approval=st.session_state.sensor_approval,
        )
    except MissionClarificationRequired as error:
        record_event(st.session_state.record_session_id, 'admission_pending',
                     {'instruction': instruction, 'feedback': rejection_message(error)})
        st.session_state.sensor_prompt = error.sensor_prompt
        st.session_state.compatibility_options = error.compatibility_options
        st.session_state.mission_alternatives = error.alternatives
        st.session_state.last_mission_transcript = error.transcript
        st.session_state.mission_smt_rejection = error.smt_rejection
        st.session_state.react_chat.append({'role': 'assistant', 'content': rejection_message(error)})
    except MissionParseError as error:
        st.session_state.react_chat.append({
            'role': 'assistant',
            'content': 'Admission could not finish: ' + str(error) + ' Send a revised instruction.',
        })
    else:
        store_verified_mission(mission, instruction)

def store_verified_mission(mission, instruction):
    missions = mission.missions if isinstance(mission, FleetMissionSpec) else [mission]
    st.session_state.nl_missions = missions
    st.session_state.last_mission_transcript = missions[0].transcript
    st.session_state.mission_alternatives = []
    st.session_state.compatibility_options = []
    st.session_state.mission_smt_rejection = {}
    st.session_state.nl_verified = True
    source = st.session_state.sensor_prompt.get('source', 'react')
    st.session_state.deployment_experimental = source == 'structured'
    st.session_state.deployment_source = source
    st.session_state.sensor_prompt = {}
    st.session_state.verified_instruction = instruction
    st.session_state.nl_plan_requested = True
    summary_rows = ['| Order | Robot | Assignment | Controller | Selection basis |',
                    '|---:|---|---|---|---|']
    for row in missions:
        robot_tasks = [mission for mission in missions if mission.carrier_id == row.carrier_id]
        order = 1 + sum(mission.sequence_index < row.sequence_index for mission in robot_tasks)
        if row.verification.get('sensor_forced'):
            basis = 'Sensor alert · user accepted RL'
        elif row.verification.get('split') == 'unseen':
            basis = 'UNSEEN layout rule'
        else:
            basis = 'Explicit ' + row.preference + ' priority'
        summary_rows.append(
            f"| {order}/{len(robot_tasks)} | {robot_label(row.carrier_id)} | "
            f"{row.dose:g} units → tumour {row.target_id} | {row.controller} | {basis} |")
    st.session_state.react_chat.append({
        'role': 'assistant', 'content': '**SMT verified. Deployment started.**\n\n' + '\n'.join(summary_rows),
    })


def ensure_deployment():
    instruction = (st.session_state.get('verified_instruction', '')
                   if st.session_state.get('deployment_source') == 'structured'
                   else react_instruction())
    admitted = bool(st.session_state.nl_verified
                    and instruction == st.session_state.get('verified_instruction')
                    and st.session_state.nl_missions)
    start_requested = bool(st.session_state.pop('nl_plan_requested', False))
    if not admitted:
        return None, False
    if start_requested or 'fleet_state' not in st.session_state:
        policies, metadata = {}, {}
        revision = model_revision()
        try:
            for mission in st.session_state.nl_missions:
                if mission.controller in POLICIES:
                    key = (mission.carrier_id, mission.sequence_index)
                    policies[key], metadata[key] = deployed_policy(
                        mission.controller, mission.target_id, mission.verification.get('policy_mode', mission.mode), mission.layout, revision)
            first = st.session_state.nl_missions[0]
            fleet = build_fleet_state(
                st.session_state.nl_missions, first.layout, first.mode, policies,
                experimental=st.session_state.get('deployment_experimental', False))
            fleet['instruction'] = st.session_state.verified_instruction
            for (carrier, sequence_index), info in metadata.items():
                state = fleet['carriers'].get(carrier)
                if state and state.get('sequence_index') == sequence_index:
                    state['policy_info'] = info
            cancel_deployment(st.session_state.get('fleet_state'))
            start_deployment(fleet, st.session_state.nl_missions, st.session_state.verified_instruction)
            st.session_state.fleet_state = fleet
        except (FileNotFoundError, ValueError) as error:
            st.error(str(error))
            return None, False
    return st.session_state.fleet_state, True


def accept_sensor_plan():
    prompt = st.session_state.sensor_prompt
    if not prompt:
        return
    approval = dict(plan_hash=prompt['proposal']['plan_hash'],
                    observation_id=prompt['observation']['observation_id'], accepted=True)
    result = verify_dispatch_smt(prompt['payload'], sensor_observation=prompt['observation'],
        sensor_approval=approval, compatibility_approvals=st.session_state.compatibility_approvals,
        urgency_authorized=bool(prompt['payload'].get('urgency_evidence')))
    if result['status'] != 'SATISFIABLE':
        st.session_state.sensor_prompt = {}
        st.session_state.mission_smt_rejection = result
        st.session_state.react_chat.append({'role': 'assistant', 'content': result['feedback']})
        return
    record_event(st.session_state.record_session_id, 'sensor_accepted', {'plan_hash': approval['plan_hash']})
    st.session_state.sensor_approval = approval
    st.session_state.react_chat.append({'role': 'user', 'content': 'I accept Q-learning Safety for the sensor-alert plan.'})
    rows = result['assignments']
    transcript = st.session_state.last_mission_transcript + [{'role': 'user', 'content': 'Observation: explicit sensor-plan acceptance; SMT reverified.'}]
    missions = [MissionSpec(row['target_id'], row['dose'], row['carrier_id'], row['layout'],
        prompt['payload']['mode'], row['preference'], row['controller'], row, transcript,
        row['sequence_index'], row['execution']) for row in rows]
    mission = missions[0] if len(missions) == 1 else FleetMissionSpec(missions,
        missions[0].layout, missions[0].mode, transcript, missions[0].execution)
    instruction = prompt.get('instruction', react_instruction()) if prompt.get('source') == 'structured' else react_instruction()
    store_verified_mission(mission, instruction)


def cancel_sensor_plan():
    cancel_deployment(st.session_state.get('fleet_state'))
    record_event(st.session_state.record_session_id, 'sensor_cancelled', {})
    st.session_state.sensor_prompt = {}
    st.session_state.sensor_approval = None
    st.session_state.mission_alternatives = []
    st.session_state.compatibility_options = []
    st.session_state.mission_smt_rejection = {}
    st.session_state.nl_verified = False
    st.session_state.nl_missions = None
    st.session_state.nl_plan_requested = False
    st.session_state.react_submit_requested = False
    st.session_state.pop('fleet_state', None)
    st.session_state.react_chat.append({'role': 'user', 'content': 'Cancel deployment.'})
    st.session_state.react_chat.append({'role': 'assistant', 'content': 'Deployment cancelled. No robot was deployed.'})


def render_plan_choices(*, compact=False):
    if st.session_state.sensor_prompt:
        for row in st.session_state.sensor_prompt['proposal']['assignments']:
            st.markdown(f"**{robot_label(row['carrier_id'])} → tumour {row['target_id']}** · {row['dose']:g} units · Q-learning Safety")
        st.button('Accept Q-learning Safety', key='accept_sensor_rl', on_click=accept_sensor_plan,
                  type='primary', use_container_width=True)
        st.button('Cancel deployment', key='cancel_sensor_rl', on_click=cancel_sensor_plan,
                  use_container_width=True)
        return

    plans = st.session_state.mission_alternatives
    if plans:
        count = len(plans)
        st.markdown(f"### Choose one of {count} verified alternative{'s' if count != 1 else ''}")
        st.caption('Each button submits exactly the plan shown on its card.')
        columns = [st.container()] if compact else st.columns(min(count, 2))
        for index, plan in enumerate(plans, start=1):
            with columns[(index - 1) % len(columns)]:
                with st.container(border=True):
                    st.markdown(f"#### Option {index} of {count}")
                    for mission in plan['assignments']:
                        reagents = ' + '.join(mission.get('reagents', [])) or 'unspecified'
                        tumour_type = mission.get('tumour_type', 'unknown').replace('_', ' ')
                        st.markdown(f"**{robot_label(mission['carrier_id'])} → tumour {mission['target_id']}**  \n"
                                    f"Dose: {mission['dose']:g} units  \n"
                                    f"Route: {mission['controller']}  \n"
                                    f"Reagent match: {reagents} → {tumour_type}")
                    if plan['status'] == 'REQUIRES_SENSOR_ACCEPTANCE':
                        st.info('SMT routing verified; sensor-alert acceptance is still required.')
                    else:
                        st.success('SMT verified')
                    if len(plan['assignments']) == 1:
                        mission = plan['assignments'][0]
                        label = (f"Choose Option {index}: {robot_label(mission['carrier_id'])} "
                                 f"→ tumour {mission['target_id']}")
                    else:
                        if plan['payload'].get('execution') == 'sequential':
                            label = f"Choose Option {index}: use {len(plan['assignments'])} ordered tasks"
                        else:
                            label = f"Choose Option {index}: deploy {len(plan['assignments'])} robots"
                    st.button(label, key=f"use_verified_{plan['plan_id']}",
                              on_click=apply_alternative_plan, args=(plan['plan_id'],),
                              type='primary', use_container_width=True)

    if st.session_state.compatibility_options:
        with st.expander('Advanced: keep the original incompatible plan'):
            st.warning('This bypasses reagent–tumour matching for this exact plan only.')
            for option in st.session_state.compatibility_options:
                required = 'X' if option['tumour_type'] == 'type_1' else 'Y'
                st.markdown(f"**{robot_label(option['carrier_id'])} → tumour {option['target_id']}**  \n"
                            f"{option['dose']:g} units · carries {' + '.join(option['reagents'])} · requires {required}")
            acknowledged = st.checkbox(
                'I understand the reagent does not match and approve this exact plan.',
                key='acknowledge_compatibility')
            st.button('Approve mismatch for original plan', disabled=not acknowledged,
                      on_click=approve_compatibility_override, use_container_width=True)


def render_chat(*, compact=False):
    st.subheader('ReAct deployment')
    st.button('Refresh conversation', on_click=reset_react_chat,
              use_container_width=True, key='refresh_react')
    if not st.session_state.react_chat:
        st.info('Enter a mission with its layout, robot, tumour, dose, and safety or speed priority.')
    for message in st.session_state.react_chat:
        with st.chat_message(message['role'], avatar=':material/science:'
                             if message['role'] == 'assistant' else ':material/person:'):
            st.markdown(message['content'])
    render_plan_choices(compact=compact)
    if st.session_state.mission_smt_rejection:
        with st.expander('SMT evidence'):
            st.json(st.session_state.mission_smt_rejection.get('diagnostics', {}))
    if st.session_state.last_mission_transcript:
        with st.expander('Thought · Action · Observation trace'):
            for index, message in enumerate(st.session_state.last_mission_transcript, start=1):
                st.markdown(f"**{index}. {'Agent' if message['role'] == 'assistant' else 'Observation'}**")
                language = 'json' if message['content'].lstrip().startswith('{') else 'text'
                st.code(message['content'], language=language)
    if not has_openai_api_key():
        st.warning('OPENAI_API_KEY was not found. Add it to `.env`, then restart Streamlit.')
    message = st.chat_input('Send a mission or reply to ReAct…', key='react_chat_input')
    if message and message.strip():
        if st.session_state.sensor_prompt and message.strip().lower().rstrip('.!') in ('accept rl', 'accept q-learning safety', 'use rl', 'i accept rl'):
            accept_sensor_plan()
            st.rerun()
        if st.session_state.sensor_prompt and message.strip().lower().rstrip('.!') in ('quit', 'cancel', 'cancel deployment'):
            cancel_sensor_plan()
            st.rerun()
        record_event(st.session_state.record_session_id, 'user_instruction', {'content': message.strip()})
        st.session_state.react_chat.append({'role': 'user', 'content': message.strip()})
        st.session_state.react_submit_requested = True
        st.rerun()


def submit_structured_mission():
    selected = [carrier for carrier in ('X', 'Y', 'Z')
                if st.session_state.get(f'structured_use_{carrier}', False)]
    if not selected:
        st.session_state.structured_error = 'Select at least one robot.'
        return
    layout = int(st.session_state.structured_layout)
    mode = st.session_state.structured_mode
    rows = []
    for carrier in selected:
        controller = st.session_state[f'structured_controller_{carrier}']
        rows.append(dict(
            carrier_id=carrier,
            target_id=st.session_state[f'structured_target_{carrier}'],
            dose=float(st.session_state[f'structured_dose_{carrier}']),
            preference='safety' if controller in ('Q-learning · Safety', 'SARSA · Safety') else 'speed',
            controller=controller,
        ))
    payload = dict(layout=layout, mode=mode, missions=rows, execution='simultaneous', experimental=True)
    observation = observe_environment(mode)
    # Stage a forced RL proposal without silently changing a structured selection.
    alert = observation['predicted_mode'] == 'very_high'
    if alert:
        for row in payload['missions']:
            row['controller'] = None
    result = verify_dispatch_smt(payload, sensor_observation=observation, planning_only=True)
    record_event(st.session_state.record_session_id, 'structured_admission',
                 {'payload': payload, 'status': result['status'], 'feedback': result.get('feedback')})
    if result['status'] == 'REQUIRES_SENSOR_ACCEPTANCE':
        st.session_state.sensor_prompt = dict(payload=payload, proposal=result, observation=observation,
            source='structured', instruction='Structured sensor-alert deployment')
        st.session_state.nl_verified = False
        st.session_state.nl_plan_requested = False
        st.session_state.mission_alternatives = []
        st.session_state.compatibility_options = []
        st.session_state.mission_smt_rejection = {}
        st.session_state.last_mission_transcript = []
        st.session_state.react_chat.append({'role': 'assistant', 'content': '**Sensor alert: very high stochasticity predicted.** A* will not be used. Accept Q-learning Safety or cancel deployment.'})
        return
    if result['status'] != 'SATISFIABLE':
        violations = result.get('diagnostics', {}).get('violations', [])
        st.session_state.structured_error = '\n'.join(v['explanation'] for v in violations) or result['feedback']
        return
    missions = [MissionSpec(
        row['target_id'], row['dose'], row['carrier_id'], layout, mode, row['preference'],
        row['controller'], row, [], row['sequence_index'], 'simultaneous')
        for row in result['assignments']]
    instruction = 'Structured deployment: ' + '; '.join(
        f"{robot_label(row.carrier_id)} delivers {row.dose:g} units to tumour {row.target_id} using {row.controller}"
        for row in missions)
    st.session_state.update(
        nl_missions=missions, nl_verified=True, verified_instruction=instruction,
        nl_plan_requested=True, deployment_experimental=True, deployment_source='structured',
        structured_error='', mission_smt_rejection={}, mission_alternatives=[],
        compatibility_options=[])


def render_structured_controls():
    with st.expander('Structured deployment'):
        st.selectbox('Layout', range(8), key='structured_layout')
        st.selectbox('Environment', ('low', 'stochastic', 'very_high', 'deterministic'),
                     format_func=lambda value: {'low': 'Low stochastic', 'stochastic': 'High stochastic',
                                                'very_high': 'Very high stochastic', 'deterministic': 'Deterministic'}[value],
                     key='structured_mode')
        for carrier in ('X', 'Y', 'Z'):
            st.checkbox('Deploy ' + robot_label(carrier), key=f'structured_use_{carrier}')
            st.selectbox(robot_label(carrier) + ' target', ('A', 'B', 'C'),
                         key=f'structured_target_{carrier}')
            st.number_input(robot_label(carrier) + ' dose', min_value=1., max_value=60., value=20.,
                            step=1., key=f'structured_dose_{carrier}')
            st.selectbox(robot_label(carrier) + ' algorithm', ('Speed · A*', *POLICIES),
                         key=f'structured_controller_{carrier}')
        st.button('Deploy structured mission', type='primary', use_container_width=True,
                  on_click=submit_structured_mission)
        if st.session_state.get('structured_error'):
            st.error(st.session_state.structured_error)


st.set_page_config(page_title='MicroDose', layout='wide')
st.session_state.setdefault('record_session_id', uuid.uuid4().hex)
if not st.session_state.get('sql_models_ready'):
    import_checkpoints(POLICY_DIR)
    st.session_state.sql_models_ready = True
title_column, guide_column = st.columns([5, 1])
title_column.title('MicroDose')
with guide_column:
    with st.popover('Grader guide', use_container_width=True):
        render_grader_guide()

for key, value in chat_defaults().items():
    st.session_state.setdefault(key, value)

if st.session_state.get('react_chat_schema_version') != 8:
    reset_react_chat()
    st.session_state.react_chat_schema_version = 8

process_pending_mission()
fleet, admitted = ensure_deployment()

if fleet is None:
    preview = snapshot_state(0)
    world_column, microscope_column = st.columns(2)
    with world_column:
        st.subheader('World Model')
        figure = plot_world_preview(preview)
        st.pyplot(figure)
        plt.close(figure)
    with microscope_column:
        st.subheader('Microscope View')
        figure = plot_microscope_view(
            preview, fleet_states=initial_robot_states(preview), highlight_target=False)
        st.pyplot(figure)
        plt.close(figure)
else:
    fleet['done'] = all(state['done'] for state in fleet['carriers'].values()) and not any(fleet['queues'].values())
    render_fleet(fleet)
    if fleet.get('needs_react'):
        st.session_state.nl_verified = False
        st.error('Release rejected. Revise the mission through ReAct.')

with st.sidebar:
    render_chat(compact=True)
    render_structured_controls()

if fleet is not None and admitted and not fleet['done'] and not fleet.get('needs_react'):
    step_fleet_state(fleet)
    sync_deployment(fleet)
    time.sleep(0.01)
    st.rerun()
