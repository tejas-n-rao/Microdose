# Clickable segmented world views for every vessel geometry.
import matplotlib.pyplot as plt
import streamlit as st

from src.simulation.live_controller import build_live_state
from src.simulation.microdose_env import LAYOUT_SEEDS, UNSEEN_LAYOUTS, vessel_label
from src.ui.live_views import plot_world_preview


def snapshot_state(layout):
    # Build a preview state. renderers show launch cells without selecting a target.
    return build_live_state(
        target_id='A', mode='low', seed=layout, dose=50,
        planner_name='Speed · A*', carrier_id='X', preference='speed',
        mission_origin='experiment')


def render_snapshot_browser():
    st.caption('Segmented vessel maps. R1, R2, and R3 mark their defined launch cells; no tumour is selected.')
    seen_tab, unseen_tab = st.tabs(['Seen layouts', 'UNSEEN layouts'])
    for tab, layouts, key in (
        (seen_tab, LAYOUT_SEEDS, 'seen_snapshot_layout'),
        (unseen_tab, UNSEEN_LAYOUTS, 'unseen_snapshot_layout'),
    ):
        with tab:
            st.session_state.setdefault(key, layouts[0])
            columns = st.columns(min(3, len(layouts)))
            for index, candidate in enumerate(layouts):
                label = ('UNSEEN ' if candidate in UNSEEN_LAYOUTS else 'Layout ') + str(candidate)
                if columns[index % len(columns)].button(
                        label, key=f'{key}_{candidate}', use_container_width=True,
                        type='primary' if st.session_state[key] == candidate else 'secondary'):
                    st.session_state[key] = candidate
            layout = st.session_state[key]
            st.markdown('**' + vessel_label(layout) + '**')
            state = snapshot_state(layout)
            figure = plot_world_preview(state)
            st.pyplot(figure)
            plt.close(figure)
            st.caption(('UNSEEN: excluded from RL training.' if layout in UNSEEN_LAYOUTS
                        else 'Seen: available during policy development.')
                       + ' The vessel contains upper, middle, and lower routes.')
