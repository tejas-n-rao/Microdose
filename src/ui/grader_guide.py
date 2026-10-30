# Collapsible reference for prompts, robots and controller choices.
import streamlit as st

from src.ui.prompt_examples import render_prompt_examples
from src.ui.robot_reference import render_robot_reference
from src.ui.snapshot_browser import render_snapshot_browser



def render_grader_guide():
    prompts, robots, dynamics, snapshots = st.tabs([
        'Prompts', 'Robots', 'Dynamics', 'Segmented world views'])

    with prompts:
        render_prompt_examples()

    with robots:
        render_robot_reference()

    with dynamics:
        st.markdown("""**How ReAct chooses the controller**

The first matching rule applies:

1. **Very-high sensor alert:** Q-learning Safety, regardless of layout or requested speed. Accept the plan or cancel.
2. **Unseen layout (5–7), no alert:** A*.
3. **Seen layout (0–4), protection before speed:** Q-learning Safety.
4. **Seen layout (0–4), speed before protection:** A*.
""")
        st.caption('ReAct asks when the required mission details are missing. SMT checks reagent compatibility, dose and capacity before deployment.')
        st.caption('Structured controls also offer SARSA Safety on seen layouts. Sensor alerts still require Q-learning Safety.')

    with snapshots:
        render_snapshot_browser()
