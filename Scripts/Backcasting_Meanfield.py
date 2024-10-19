import itertools
import numpy as np
from collections import deque
import matplotlib.pyplot as plt

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
final_states = [0.20, 0.15, 0.06]  # Desired final biking modal shares

# Action definitions
class Action:
    def __init__(self, name, effects=None, cooldown=None, preparation=None, one_time=False,
                 success_probability=1.0, conditions=None, interactions=None, costs=None):
        self.name = name
        self.effects = self._normalize_effects(effects) if effects is not None else {}
        self.cooldown = cooldown if cooldown is not None else 0
        self.one_time = one_time
        self.costs = costs if costs is not None else {}
        self.preparation = preparation
        self.success_probability = success_probability
        self.conditions = conditions if conditions is not None else {}
        self.interactions = interactions if interactions is not None else {}

    def _normalize_effects(self, effects):
        # Ensure that the sum of all effects is zero
        total_effect = sum(effects.values())
        if total_effect != 0:
            # Distribute the total effect equally among unspecified modes
            unspecified_modes = set(['s', 'c', 'p', 'o']) - set(effects.keys())
            if unspecified_modes:
                adjustment = -total_effect / len(unspecified_modes)
                for mode in unspecified_modes:
                    effects[mode] = adjustment
            else:
                # If all modes are specified, adjust the 'o' mode
                effects['o'] = effects.get('o', 0) - total_effect
        return effects

    def evaluate_success_probability(self, executed_actions):
        probability = self.success_probability
        for interaction_action, adjustment in self.interactions.items():
            if executed_actions.get(interaction_action, 0) == 1:
                probability += adjustment
        return min(max(probability, 0.0), 1.0)

    def calculate_cost(self, executed_actions):
        cost = self.costs.get('base', 0)
        for interaction_action, adjustment in self.costs.get('interactions', {}).items():
            if executed_actions.get(interaction_action, 0) == 1:
                cost *= (1 + adjustment)
        return max(cost, 0)  # Ensure cost is non-negative

# Define the actions with their parameters
ACTIONS = {
    # Increase Policy Coordination 
    'IPC': Action(
        name='IPC',
        cooldown=2,
        costs={'base': 0.01}
    ),
    # Anti-Car Propoganda 
    'ACP': Action(
        name='ACP',
        effects={'s': 0.01, 'c': -0.005},
        cooldown=1,
        costs={'base': 0.01}
    ),
    # Raise Prices Public Transport 
    'RPPT': Action(
        name='RPPT',
        effects={'p': -0.005, 's': 0.0025, 'c': 0.0025},
        one_time=True,
        success_probability=0.2,
        interactions={
            'IPC': 0.2,
            'LBF': 0.2
        },
        costs={
            'base': 0.1,
            'interactions': {
                'LBF': -0.5,  # LBF reduces cost by 50%
                'IPC': -0.4   # IPC reduces cost by 40%
            }
        }
    ),
    # Invest in Biking Infrastructure
    'IBI': Action(
        name='IBI',
        effects={'s': 0.02, 'c': -0.01},
        one_time=True,
        success_probability=0.2,
        interactions={
            'IPC': 0.2,
            'LBF': 0.2
        },
        costs={
            'base': 0.1,
            'interactions': {
                'LBF': -0.5,  # LBF reduces cost by 50%
                'IPC': -0.4   # IPC reduces cost by 40%
            }
        }
    ),
    # Lobby for Biking Finance 
    'LBF': Action(
        name='LBF',
        cooldown=1,
        success_probability=0.2,
        interactions={
            'ACP': 0.2,
        },
        costs={'base': 0.01}
    ),
    #Levy Taxes on Car Parking 
    'LTP': Action(
        name='LTP',
        effects={'c': -0.007, 's': 0.0035, 'p': 0.0035},
        one_time=True,
        success_probability=0.2,
        interactions={
            'ACP': 0.2,
            'IPC': 0.2,
        },
        costs={
            'base': 0.01,
            'interactions': {
                'LBF': -0.5,  # LBF reduces cost by 50%
                'IPC': -0.4   # IPC reduces cost by 40%
            }
        }

    )
}

MEAN_FIELD_EFFECTS = {
    'base': {
        'effects': {'c': 0.000, 'p': 0.000},
        'probability': 0.8  # Base probability of the effect occurring
    },
    'reactions': {
        'ACP': {
            'effects': {'c': 0.002},
            'probability': 0.7
        },
        'LTP': {
            'effects': {'c': 0.002, 'p': 0.002},
            'probability': 0.6
        },
        'RPPT': {
            'effects': {'c': 0.0025, 's': -0.0025},
            'probability': 0.5
        }
    }
}


# Utility function (for example purposes)
def utility_function(path):
    target_biking_share = min(final_states)
    total_utility = 0
    total_cost = 0
    for step in path:
        modal_shares = step['modal_shares']
        step_utility = -((modal_shares['s'] - target_biking_share) ** 2)
        total_utility += step_utility
        
        # Calculate cost for this step
        step_cost = sum(ACTIONS[DECISIONS[i]].costs['base'] for i, d in enumerate(step['decisions']) if d == 1)
        total_cost += step_cost
    
    # Normalize utility and cost to be on similar scales
    normalized_utility = total_utility / len(path)
    normalized_cost = total_cost / len(path)
    
    # Combine utility and cost (you can adjust the weights as needed)
    combined_score = normalized_utility - normalized_cost
    
    return combined_score

# Exogenous factors (if any)
EXOGENOUS_FACTORS = {
    1: { # Bike Technology : EBike, Cargo Bike, Folding Bike, Shared Bikes 
        'effects': {'s': 0.01, 'c': -0.005, 'p': -0.0025, 'o': -0.0025},
        'probability': 0.7,
        'label': 'E-Bikes'
    }, 
    2: { # Pandemic hits
        'effects': {'s': 0.01, 'c': -0.005, 'p': -0.0025, 'o': -0.0025},
        'probability': 0.01,
        'label': 'Pandemic'
    },
    3: { # EU Biking Directive passed
        'effects': {'s': 0.05, 'c': -0.025, 'p': -0.0125, 'o': -0.0125},
        'probability': 0.05,
        'label': 'EU Biking Directive'
    },
    4: { #Shared Bikes + Cargo Bikes + Folding Bikes 
        'effects': {'s': 0.01, 'p': -0.002, 'c': -0.004, 'o': -0.004},
        'probability': 0.8, 
        'label': 'Shared Biking'
    },
    5: {
        'effects': {'s': 0.5, 'p': 0.3, 'c': -0.4, 'o': -0.4},
        'probability': 0.0000000001,
        'label': 'Tipping point Social Perception'
    },
}


# Modify the state_transition function
def state_transition(modal_shares_prev, decisions, current_time):
    modal_shares = modal_shares_prev.copy()
    executed_actions = {action_name: decisions[DECISION_INDICES[action_name]] for action_name in DECISIONS}
    actions_to_disable = set()
    total_cost = 0
    debug_log = []

    for action_name, action in ACTIONS.items():
        if executed_actions.get(action_name, 0):
            success_prob = action.evaluate_success_probability(executed_actions)
            success = np.random.rand() <= success_prob
            if success:
                debug_log.append(f"Action {action_name} succeeded:")
                for mode, delta in action.effects.items():
                    modal_shares[mode] += delta
                    debug_log.append(f"  {mode}: {delta:+.4f}")
                total_cost += action.calculate_cost(executed_actions)
            else:
                debug_log.append(f"Action {action_name} failed")
                if action.one_time:
                    actions_to_disable.add(action_name)

    mf_effects = mean_field_response(modal_shares_prev, decisions)
    debug_log.append("Mean Field Effects:")
    for mode, delta in mf_effects.items():
        modal_shares[mode] += delta
        debug_log.append(f"  {mode}: {delta:+.4f}")

    # Exogenous factors
    exogenous_effects = EXOGENOUS_FACTORS.get(current_time, {})
    if exogenous_effects:
        debug_log.append(f"Exogenous Effects ({exogenous_effects['label']}):")
        if np.random.random() < exogenous_effects['probability']:
            total_exogenous_effect = sum(exogenous_effects['effects'].values())
            adjustment = -total_exogenous_effect / (len(modal_shares) - len(exogenous_effects['effects']))
            for mode in modal_shares:
                if mode in exogenous_effects['effects']:
                    delta = exogenous_effects['effects'][mode]
                else:
                    delta = adjustment
                modal_shares[mode] += delta
                debug_log.append(f"  {mode}: {delta:+.4f}")
        else:
            debug_log.append(f"  No {exogenous_effects['label']} effects applied (probability not met)")

    # Ensure modal shares are within [0,1]
    for mode in modal_shares:
        if modal_shares[mode] < 0 or modal_shares[mode] > 1:
            debug_log.append(f"Clipping {mode} from {modal_shares[mode]:.4f} to {min(max(modal_shares[mode], 0), 1):.4f}")
        modal_shares[mode] = min(max(modal_shares[mode], 0), 1)

    # Normalize modal shares to sum to 1
    total_share = sum(modal_shares.values())
    if not np.isclose(total_share, 1, atol=1e-6):
        debug_log.append("Adjusting modal shares to sum to 1:")
        adjustment_factor = 1 / total_share
        for mode in modal_shares:
            old_share = modal_shares[mode]
            modal_shares[mode] *= adjustment_factor
            debug_log.append(f"  {mode}: {old_share:.4f} -> {modal_shares[mode]:.4f}")

    return modal_shares, actions_to_disable, total_cost, debug_log
