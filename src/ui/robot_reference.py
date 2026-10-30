# Compact reference for all robots, independent of current deployment.
import streamlit as st
from src.robot_identity import ROBOT_LABELS
from src.verification.smt_verifier import RULES



def render_robot_reference():
    st.subheader('Delivery robots')

    for column, (carrier, label) in zip(st.columns(3), ROBOT_LABELS.items()):
        reagents = RULES['carrier_reagents'][carrier]
        targets = [target for target, kind in RULES['tumour_types'].items()
                   if set(reagents) & set(RULES['compatible_reagents'][kind])]

        with column:
            st.markdown(f"### {label}")
            st.markdown(f"**Maximum capacity:** {RULES['carrier_capacities'][carrier]} units")
            st.markdown('**Reagents:** ' + ' + '.join(reagents))
            st.markdown('**Compatible tumours:** ' + ', '.join(targets))
    st.caption('Reagent X → type 1 (tumours A/B). Reagent Y → type 2 (tumour C). Robot R3 carries both. Robot names and reagent names are separate labels. Only requested robots are deployed.')
