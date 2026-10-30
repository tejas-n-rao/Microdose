# Compact shared-clock deployment view.
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src.robot_identity import robot_label
from src.simulation.microdose_env import GRID
from src.ui.live_views import plot_microscope_view, world_model_frame, healthy_marker_cells, ROBOT_COLORS


def plot_fleet_world(fleet):
    # Fleet overlay on the canonical muted-purple/rust world schema.
    states = fleet['carriers']
    first = next(iter(states.values()))
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.imshow(world_model_frame(first), origin='upper')

    for index, (x, y) in enumerate(healthy_marker_cells(first['seed'], first['blocked'])):
        ax.add_patch(plt.Rectangle(
            (x - .5, y - .5), 1, 1, facecolor='#9b6b38', edgecolor='#f7e8c9',
            linewidth=.7, alpha=.82, label='healthy tissue risk' if index == 0 else None,
            zorder=3))

    for target, tumour in first['tumours'].items():
        x, y = tumour['true']
        ax.scatter(x, y, marker='s', s=165, c='#ff2f6d', edgecolors='white', zorder=15)
        ax.annotate(target, (x, y), ha='center', va='center', color='white', weight='bold', zorder=16)

    for carrier, state in states.items():
        plan = state['plan'].path
        if plan and not state['done']:
            ax.plot([p[0] for p in plan], [p[1] for p in plan], color='white', linestyle=':',
                    linewidth=1.4, label=robot_label(carrier) + ' reference', zorder=9)
        recovery = state.get('recovery_path', [])
        if recovery and not state['done']:
            ax.plot([p[0] for p in recovery], [p[1] for p in recovery], color='#2ec4b6',
                    linestyle='--', linewidth=2, label=robot_label(carrier) + ' recovery', zorder=10)
        trace = state['actual_trace']
        ax.plot([p[0] for p in trace], [p[1] for p in trace], color=ROBOT_COLORS[carrier],
                linewidth=2.2, label=robot_label(carrier) + ' trace', zorder=11)
        ax.scatter(*state['actual'], color=ROBOT_COLORS[carrier], edgecolors='white', s=110, zorder=13)
        ax.scatter(*state['estimated'], edgecolors='#2ec4b6', facecolors='none', s=190,
                   linewidths=2, zorder=12)
        ax.annotate(robot_label(carrier), state['actual'], xytext=(5, -8),
                    textcoords='offset points', color='white', weight='bold', zorder=14)
        if state.get('release_event'):
            ax.scatter(*state['release_event']['position'], marker='s', s=190,
                       facecolors='none', edgecolors='#35ff69', linewidths=2, zorder=17)

    ax.set_title('World Model · layout ' + str(fleet['seed']))
    ax.set_xlim(-.5, GRID - .5)
    ax.set_ylim(GRID - .5, -.5)
    tick_step = 4 if GRID > 28 else 2 if GRID > 20 else 1
    ax.set_xticks(range(0, GRID, tick_step))
    ax.set_yticks(range(0, GRID, tick_step))
    ax.legend(loc='upper left', fontsize=7)
    fig.tight_layout()
    return fig


def task_order_labels(fleet):
    # Count each robot's tasks, rather than the shared mission IDs.
    indices = {carrier: {state.get('sequence_index', 0)}
               for carrier, state in fleet['carriers'].items()}
    for task in fleet.get('completed_tasks', []):
        indices[task['robot']].add(task['sequence_index'])
    for carrier, queued in fleet.get('queues', {}).items():
        indices[carrier].update(mission.sequence_index for mission in queued)
    return {(carrier, index): f"{position}/{fleet['sequence_totals'][carrier]}"
            for carrier, tasks in indices.items()
            for position, index in enumerate(sorted(tasks), start=1)}


def render_fleet(fleet):
    states = fleet['carriers']
    first = next(iter(states.values()))

    world_column, microscope_column = st.columns(2)
    with world_column:
        st.subheader('World Model')
        figure = plot_fleet_world(fleet)
        st.pyplot(figure)
        plt.close(figure)
    with microscope_column:
        st.subheader('Microscope View')
        base = dict(first)
        base['flow'] = base['world'].flows[fleet['tick']].tolist()
        figure = plot_microscope_view(base, fleet_states=states)
        st.pyplot(figure)
        plt.close(figure)

    columns = st.columns(4)
    columns[0].metric('Timestep', f"{fleet['tick']} / {fleet['max_steps']}")
    columns[1].metric('Robots', len(states))
    columns[2].metric('Delivered', sum(state['terminal'] == 'delivery' for state in states.values()))
    columns[3].metric('Status', 'Complete' if fleet['done'] else 'Running')

    orders = task_order_labels(fleet)
    rows = [dict(
        order=orders[carrier, state.get('sequence_index', 0)],
        robot=robot_label(carrier),
        instruction=f"{state['dose']:g} units → tumour {state['target_id']}",
        reagent='+'.join(state['mission_verification']['reagents']),
        algorithm=state['planner_name'],
        safety=round(state['safety_score'], 1),
        outcome=state['terminal'] or 'active',
    ) for carrier, state in states.items()]
    rows.extend(dict(
        order=orders[task['robot'], task['sequence_index']],
        robot=robot_label(task['robot']),
        instruction=f"{task['dose']:g} units → tumour {task['target_id']}",
        reagent='completed', algorithm=task['controller'], safety=round(task['safety_score'], 1),
        outcome=task['terminal']) for task in fleet.get('completed_tasks', []))
    for carrier, queued in fleet.get('queues', {}).items():
        for mission in queued:
            rows.append(dict(
                order=orders[carrier, mission.sequence_index],
                robot=robot_label(carrier),
                instruction=f"{mission.dose:g} units → tumour {mission.target_id}",
                reagent='pending', algorithm=mission.controller, safety=None, outcome='waiting'))
    rows.sort(key=lambda row: (row['robot'], row['order']))
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    with st.expander('Deployment details'):
        if fleet.get('instruction'):
            st.code(fleet['instruction'], language=None, wrap_lines=True)
        st.json({
            'layout': fleet['seed'],
            'sensor_prediction': (fleet['admission']['diagnostics'].get('sensor_observation') or {}).get('predicted_mode', 'not sampled'),
            'sensor_forced_rl': fleet['admission']['diagnostics'].get('sensor_forced', False),
            'execution': fleet['admission']['diagnostics'].get('execution', 'simultaneous'),
            'urgent_tumour': fleet['admission']['diagnostics']['urgent_target'],
            'smt_status': fleet['admission']['status'],
        })
