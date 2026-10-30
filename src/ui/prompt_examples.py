# Short copyable prompts available before any mission is admitted.
import streamlit as st

EXAMPLES = (
    ('Layout 1: lower-corridor detour', 'For Layout 1, Deploy R1 to B safely, and R3 to B fast. Deliver 20 units each.', 'R1: Q-learning Safety via the lower corridor. R3: A* through the middle. Both delivered in the structured low-stochasticity replay, with less observed tissue exposure for R1.'),
    ('Layout 4: avoid the tissue cluster', 'For Layout 4, Deploy R3 to B safely, and R1 to B fast. Deliver 20 units each.', 'R3: Q-learning Safety along the upper edge of the middle route. R1: direct A*. Both delivered in the structured deterministic replay, with less observed tissue exposure for R3.'),
    ('Sequential tasks for one robot', 'Deliver 20 units to tumour A via R1 and then 25 units to tumour B in layout 0. Protect healthy tissue for both tasks.', 'Two ordered R1 tasks. Q-learning Safety handles both tasks; combined payload remains below 60 units.'),
    ('Sensor restriction', 'On layout 0, robot R1 delivers 40 units to A. Arrive as quickly as possible.', 'If sensors predict very high: A* unavailable. Accept Q-learning Safety or cancel. Without the alert: A*.'),
    ('Protect healthy tissue', 'On layout 0, robot R1 delivers 50 units to A. Avoid damaging healthy tissue even if the trip takes longer.', 'Q-learning Safety on this seen layout.'),
    ('Three robots, speed routes', 'On layout 1, deploy simultaneously: robot R1 delivers 50 units to A, robot R2 delivers 40 units to C, and robot R3 delivers 55 units to B. Arrive as quickly as possible for all robots.', 'A* for all three; execution continues until every assigned robot finishes.'),
    ('Different priorities', 'On layout 2, deploy simultaneously: robot R1 delivers 50 units to A and protects healthy tissue even if slower; robot R2 delivers 40 units to C and arrives as quickly as possible.', 'R1 uses Q-learning Safety; R2 uses A*.'),
    ('Unseen vessel', 'On layout 5, robot R3 delivers 50 units to C. Protect healthy tissue even if the trip takes longer.', 'UNSEEN label; A* required.'),
    ('Reagent mismatch', 'On layout 3, robot R1 delivers 40 units to C. Arrive as quickly as possible.', 'Reject; choose a compatible robot or explicitly approve this exact mismatch.'),
    ('Excess dose', 'On layout 4, robot R1 delivers 60 units to A. Arrive as quickly as possible.', 'Reject; A has a 50-unit target budget. Revise the dose.'),
    ('Second unseen vessel', 'On layout 6, robot R2 delivers 45 units to C. Arrive as quickly as possible.', 'UNSEEN label; A* with a compatible reagent.'),
    ('Third unseen vessel', 'On layout 7, robot R1 delivers 50 units to A. Protect healthy tissue even if the trip takes longer.', 'UNSEEN label; SMT still requires A* because the geometry was excluded from RL training.'),
)



def render_prompt_examples():
    st.caption('Some prompts to test edge cases:')
    st.caption('Route examples use the structured settings stated below. Natural-language flow is sampled, so traces can vary. A very-high sensor alert requires accepted Q-learning Safety for every robot or cancellation.')

    for title, prompt, expected in EXAMPLES:
        st.markdown('**' + title + '**')
        st.code(prompt, language=None, wrap_lines=True)
        st.caption('Expected: ' + expected)
