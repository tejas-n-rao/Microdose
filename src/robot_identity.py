# Robot display names. legacy carrier IDs remain stable for SMT and saved data.
ROBOT_LABELS = {'X': 'R1', 'Y': 'R2', 'Z': 'R3'}
ROBOT_CARRIERS = {label: carrier for carrier, label in ROBOT_LABELS.items()}


def robot_label(carrier):
    return ROBOT_LABELS.get(carrier, carrier or 'Robot')
