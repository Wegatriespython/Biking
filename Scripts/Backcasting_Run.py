from Backcasting_Meanfield import *

import matplotlib.pyplot as plt
import cProfile
import pstats
from io import StringIO
import numpy as np
import random
def set_random_seed(seed=42):
    np.random.seed(seed)
    random.seed(seed)
# ------------------------- End of User Input Section -------------------------
# State node representation
class StateNode:
    def __init__(self, time, modal_shares, decisions=None, parent=None,
                 cooldowns=None, disabled_actions=None, cost=0, debug_log=None):
        self.time = time
        self.modal_shares = modal_shares.copy()
        self.decisions = decisions
        self.parent = parent
        self.children = []
        self.cost = cost
        self.debug_log = debug_log

        if cooldowns is None:
            self.cooldowns = {action: 0 for action in DECISIONS}
        else:
            self.cooldowns = cooldowns.copy()
        if disabled_actions is None:
            self.disabled_actions = set()
        else:
            self.disabled_actions = disabled_actions.copy()
    def __repr__(self):
        modal_shares_str = ', '.join([f"{k}: {v:.3f}" for k, v in self.modal_shares.items()])
        return (f"Time: {self.time}, {modal_shares_str}, Decisions: {self.decisions}")
# Mean-field approximation for other players' response
def mean_field_response(modal_shares, decisions):
    mf_effects = {'s': 0, 'c': 0, 'p': 0, 'o': 0}
    
    # Apply base effects
    if np.random.random() < MEAN_FIELD_EFFECTS['base']['probability']:
        for mode, delta in MEAN_FIELD_EFFECTS['base']['effects'].items():
            mf_effects[mode] += delta
    
    # Apply reaction effects based on our decisions
    for action_name, reaction in MEAN_FIELD_EFFECTS['reactions'].items():
        idx = DECISION_INDICES[action_name]
        if decisions[idx]:
            if np.random.random() < reaction['probability']:
                for mode, delta in reaction['effects'].items():
                    mf_effects[mode] += delta
    
    return mf_effects

# State transition function
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
            unspecified_modes = set(modal_shares.keys()) - set(exogenous_effects['effects'].keys())
            
            if unspecified_modes:
                adjustment = -total_exogenous_effect / len(unspecified_modes)
            else:
                adjustment = 0  # No adjustment needed if all modes are specified
            
            for mode in modal_shares:
                if mode in exogenous_effects['effects']:
                    delta = exogenous_effects['effects'][mode]
                elif unspecified_modes:
                    delta = adjustment
                else:
                    delta = 0  # No change for modes not specified when all modes are covered
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

def build_graph(initial_modal_shares, final_states):
    root = StateNode(time=0, modal_shares=initial_modal_shares)
    queue = deque([root])
    paths = []
    while queue:
        current_node = queue.popleft()
        if current_node.time == len(TIME_PERIODS):
            if np.isclose(current_node.modal_shares['s'], final_states, atol=0.005).any():
                path = []
                node = current_node
                while node.parent is not None:
                    path.append({
                        'time': node.time,
                        'decisions': node.decisions,
                        'modal_shares': node.modal_shares,
                        'cost': node.cost,
                        'debug_log': node.debug_log
                    })
                    node = node.parent
                path.reverse()
                paths.append(path)
            continue
        # Decrease cooldowns
        cooldowns_next = {action: max(current_node.cooldowns[action] - 1, 0) for action in DECISIONS}
        # Determine available actions
        available_actions = [action for action in DECISIONS
                             if cooldowns_next[action] == 0 and action not in current_node.disabled_actions]

       # Generate all possible combinations of available actions (power set)
        action_combinations = []
        for r in range(len(available_actions) + 1):
            action_combinations.extend(itertools.combinations(available_actions, r))
        # For each combination, create the decision vector
        for action_subset in action_combinations:
            decisions = [0] * len(DECISIONS)
            for action in action_subset:
                idx = DECISION_INDICES[action]
                decisions[idx] = 1

            modal_shares_next, actions_to_disable, step_cost, debug_log = state_transition(
                current_node.modal_shares,
                decisions,
                current_node.time  # Pass the current time to state_transition
            )
            # Update cooldowns and disabled actions
            cooldowns_updated = cooldowns_next.copy()
            disabled_actions_updated = current_node.disabled_actions.copy()
            # Set cooldowns and disable actions as needed
            for action in action_subset:
                action_obj = ACTIONS[action]
                idx = DECISION_INDICES[action]
                # Set cooldown if action has a cooldown
                if action_obj.cooldown is not None:
                    cooldowns_updated[action] = action_obj.cooldown
                # Disable one-time actions
                if action_obj.one_time:
                    disabled_actions_updated.add(action)
            # Disable actions that failed
            disabled_actions_updated.update(actions_to_disable)
            # Create child node
            child_node = StateNode(
                time=current_node.time + 1,
                modal_shares=modal_shares_next,
                decisions=decisions,
                parent=current_node,
                cooldowns=cooldowns_updated,
                disabled_actions=disabled_actions_updated,
                cost=current_node.cost + step_cost,
                debug_log=debug_log
            )
            current_node.children.append(child_node)
            queue.append(child_node)

    return paths

# Execute the graph building
paths = build_graph(initial_modal_shares, final_states)

# Display the results
print(f"Total feasible paths to desired final states: {len(paths)}\n")
# Find and visualize the most parsimonious path

def evaluate_path(path):
    utility = utility_function(path)
  # Negative because we want to maximize utility but minimize decisions
    return utility 

def find_best_path(paths):
    if not paths:
        return None
    return max(paths, key=evaluate_path)

def visualize_best_path(path):
    times = [0] + [step['time'] for step in path]
    shares = [initial_modal_shares['s']] + [step['modal_shares']['s'] for step in path]
    fig, ax = plt.subplots(figsize=(15, 8))
    ax.plot(times, shares, marker='o')
    
    for i, step in enumerate(path):
        y_pos = step['modal_shares']['s']
        
        annotations = []
        successful_actions = []
        failed_actions = []
        exogenous_effect = None
        
        for log_entry in step['debug_log']:
            if "succeeded" in log_entry:
                action = log_entry.split()[1]
                successful_actions.append(action)
            elif "failed" in log_entry:
                action = log_entry.split()[1]
                failed_actions.append(action)
            elif "Exogenous Effects" in log_entry:
                exogenous_effect = log_entry.split(":")[0]  # Capture the label
            elif "s:" in log_entry and exogenous_effect:
                # This means exogenous effects were applied
                annotations.append(exogenous_effect)
                exogenous_effect = None  # Reset to avoid duplicate entries
        
        if successful_actions:
            annotations.append(f"Successful: {', '.join(successful_actions)}")
        if failed_actions:
            annotations.append(f"Failed: {', '.join(failed_actions)}")
        
        annotation_text = '\n'.join(annotations)
        
        if annotation_text:
            ax.annotate(annotation_text, (step['time'], y_pos), 
                        textcoords="offset points", xytext=(0,10), ha='center', va='bottom',
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
                        wrap=True)
        
        # Color-code the actions on the x-axis
        action_text = []
        if successful_actions:
            action_text.append(ax.annotate(', '.join(successful_actions), (step['time'], 0), 
                               xytext=(0, -20), textcoords="offset points", ha='center', va='top',
                               color='green', fontweight='bold'))
        if failed_actions:
            action_text.append(ax.annotate(', '.join(failed_actions), (step['time'], 0), 
                               xytext=(0, -35), textcoords="offset points", ha='center', va='top',
                               color='red', fontweight='bold'))
        
        # Adjust text position if overlapping
        if len(action_text) > 1:
            action_text[1].set_position((step['time'], -50))
    
    ax.set_xlabel('Time Period')
    ax.set_ylabel('Biking Modal Share')
    ax.set_title('Best Path for Increasing Biking Modal Share')
    
    # Add legend
    legend_text = '\n'.join([f"{abbr}: {name}" for abbr, name in zip(DECISIONS, [       
        "Improve Policy Coordination",
        "Anti-Car Propaganda",
        "Raise Prices Public Transport",
        "Invest in Bike Infrastructure",
        "Lobby for Biking Finance",
        "Levy Tax on Car Parking"
    ])])
    
    ax.text(1.05, 0.5, legend_text, transform=ax.transAxes, 
            bbox=dict(facecolor='white', edgecolor='black', alpha=0.7),
            verticalalignment='center')
    
    plt.tight_layout()
    plt.show()
best_path = find_best_path(paths)

# Wrap the main execution in a function
def main():
    global paths  # Make paths global so it can be accessed outside the function
    # Set the random seed at the beginning of main
    set_random_seed()
    
    paths = build_graph(initial_modal_shares, final_states)

    print(f"Total feasible paths to desired final states: {len(paths)}\n")

    best_path = find_best_path(paths)

    if best_path:
        print("Best Path:")
        for step in best_path:
            decision_names = [DECISIONS[i] for i, d in enumerate(step['decisions']) if d == 1]
            modal_shares_str = ', '.join([f"{k.upper()}: {v:.3f}" for k, v in step['modal_shares'].items()])
            print(f"Time {step['time']}:")
            print(f"  Decisions: {decision_names}")
            print(f"  Modal Shares: {modal_shares_str}")
            print(f"  Cost: {step['cost']:.3f}")
            print("  Debug Log:")
            for log_entry in step['debug_log']:
                print(f"    {log_entry}")
            print()
        visualize_best_path(best_path)
    else:
        print("No feasible paths found.")

# Add this at the end of the file
if __name__ == "__main__":
    # Run the profiler
    profiler = cProfile.Profile()
    profiler.enable()

    main()

    profiler.disable()
    s = StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
    ps.print_stats(20)  # Print top 20 time-consuming functions
    print(s.getvalue())

def set_random_seed(seed=42):
    np.random.seed(seed)
    random.seed(seed)
