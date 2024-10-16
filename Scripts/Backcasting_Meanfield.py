import itertools
import numpy as np
from collections import deque

# ------------------------- User Input Section -------------------------

# Time periods
TIME_PERIODS = [1, 2, 3, 4]

# Define actions and their indices
DECISIONS = ['IPC', 'ACP', 'RPPT', 'IBI', 'LBF', 'LTP']
DECISION_INDICES = {name: idx for idx, name in enumerate(DECISIONS)}

# Initial modal shares (should sum to 1)
initial_modal_shares = {
    's': 0.01,  # Initial biking modal share
    'c': 0.50,  # Initial car modal share
    'p': 0.40,  # Initial public transport modal share
    'o': 0.09   # Other transport modes (calculated to sum to 1)
}

# Desired final biking modal shares
final_states = [0.20, 0.07, 0.04]  # Desired final biking modal shares

# Action definitions
class Action:
    def __init__(self, name, effects=None, cooldown=None, preparation=None, one_time=False,
                 success_probability=1.0, conditions=None, interactions=None):
        self.name = name
        self.effects = effects if effects is not None else {}  # Effects on modal shares
        self.cooldown = cooldown          # Cooldown periods after action execution
        self.preparation = preparation    # Preparation periods before action takes effect
        self.one_time = one_time          # Whether the action can be taken only once
        self.success_probability = success_probability  # Base success probability
        self.conditions = conditions if conditions is not None else {}  # Conditions for success
        self.interactions = interactions if interactions is not None else {}  # Interactions with other actions

    def evaluate_success_probability(self, executed_actions):
        # Evaluate success probability based on conditions and interactions
        probability = self.success_probability
        for condition_action, required_state in self.conditions.items():
            if executed_actions.get(condition_action, 0) != required_state:
                probability = 0.0
        for interaction_action, adjustment in self.interactions.items():
            if executed_actions.get(interaction_action, 0) == 1:
                probability += adjustment
        # Ensure probability is within [0, 1]
        probability = min(max(probability, 0.0), 1.0)
        return probability

# Define the actions with their parameters
ACTIONS = {
    'IPC': Action(
        name='IPC',
        cooldown=2
    ),
    'ACP': Action(
        name='ACP',
        effects={'s': 0.01, 'c': -0.005},
        cooldown=1
    ),
    'RPPT': Action(
        name='RPPT',
        effects={'p': -0.005, 's': -0.0025, 'c': -0.0025},
        one_time=True
    ),
    'IBI': Action(
        name='IBI',
        effects={'s': 0.02, 'c': -0.01},
        one_time=True,
        success_probability=0.2,
        interactions={
            'IPC': 0.4,  # Increases success probability by 0.4 if IPC is executed
            'LBF': 0.4   # Increases success probability by 0.4 if LBF is executed
        }
    ),
    'LBF': Action(
        name='LBF',
        cooldown=1
    ),
    'LTP': Action(
        name='LTP',
        effects={'c': -0.007, 's': -0.0035, 'p': -0.0035},
        one_time=True
    )
}

# Mean-field approximation parameters
MEAN_FIELD_EFFECTS = {
    'base': {'c': 0.005, 'p': -0.003},  # Base adjustments to modal shares
    'reactions': {  # Adjustments based on our actions
        'ACP': {'c': 0.002},
        'LTP': {'c': 0.002}
    }
}

# Utility function (for example purposes)
def utility_function(modal_shares):
    # Define the utility based on modal shares
    target_biking_share = 0.2
    utility = -((modal_shares['s'] - target_biking_share) ** 2)
    return utility

# Exogenous factors (if any)
EXOGENOUS_FACTORS = {
    # Define exogenous effects on modal shares (e.g., seasonal effects)
    # Time period as key, adjustments as values
    # Example: {2: {'s': 0.01, 'c': -0.005}}
}

