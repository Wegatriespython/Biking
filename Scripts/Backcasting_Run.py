from Backcasting_Meanfield import *
import matplotlib.pyplot as plt

# ------------------------- End of User Input Section -------------------------
# State node representation
class StateNode:
    def __init__(self, time, modal_shares, decisions=None, parent=None,
                 cooldowns=None, disabled_actions=None):
        self.time = time
        self.modal_shares = modal_shares.copy()
        self.decisions = decisions  # Decisions taken to reach this state
        self.parent = parent        # Parent StateNode
        self.children = []          # Child StateNodes

        # Cooldowns: mapping action name to remaining cooldown periods
        if cooldowns is None:
            self.cooldowns = {action: 0 for action in DECISIONS}
        else:
            self.cooldowns = cooldowns.copy()

        # Disabled actions: set of actions that are disabled
        if disabled_actions is None:
            self.disabled_actions = set()
        else:
            self.disabled_actions = disabled_actions.copy()

    def __repr__(self):
        modal_shares_str = ', '.join([f"{k}: {v:.3f}" for k, v in self.modal_shares.items()])
        return (f"Time: {self.time}, {modal_shares_str}, Decisions: {self.decisions}")

# Mean-field approximation for other players' response
def mean_field_response(modal_shares, decisions):
    # Start with base effects
    mf_effects = MEAN_FIELD_EFFECTS['base'].copy()
    # Adjustments based on our decisions
    for action_name, adjustments in MEAN_FIELD_EFFECTS['reactions'].items():
        idx = DECISION_INDICES[action_name]
        if decisions[idx]:
            for mode, delta in adjustments.items():
                mf_effects[mode] = mf_effects.get(mode, 0) + delta
    return mf_effects

# State transition function
def state_transition(modal_shares_prev, decisions):
    modal_shares = modal_shares_prev.copy()
    D = decisions

    # Map decisions to action names for easy access
    executed_actions = {action_name: D[DECISION_INDICES[action_name]] for action_name in DECISIONS}

    # Actions that need to be disabled (e.g., failed actions)
    actions_to_disable = set()

    # Apply action effects
    for action_name, action in ACTIONS.items():
        if executed_actions.get(action_name, 0):
            # Evaluate success probability
            success_prob = action.evaluate_success_probability(executed_actions)
            success = np.random.rand() <= success_prob  # Simulate success
            if success:
                for mode, delta in action.effects.items():
                    modal_shares[mode] += delta
            else:
                # If action fails and is one-time, disable it
                if action.one_time:
                    actions_to_disable.add(action_name)

    # Mean-field response from other players
    mf_effects = mean_field_response(modal_shares_prev, decisions)
    for mode, delta in mf_effects.items():
        modal_shares[mode] += delta

    # Exogenous factors
    exogenous_effects = EXOGENOUS_FACTORS.get(len(TIME_PERIODS), {})
    for mode, delta in exogenous_effects.items():
        modal_shares[mode] += delta

    # Ensure modal shares sum to 1 and are within [0,1]
    total_share = sum(modal_shares.values())
    if total_share != 1:
        # Adjust 'other' share to balance the total to 1
        modal_shares['o'] = 1 - sum([modal_shares[mode] for mode in ['s', 'c', 'p']])

    # Clip modal shares to [0,1]
    for mode in modal_shares:
        modal_shares[mode] = min(max(modal_shares[mode], 0), 1)

    return modal_shares, actions_to_disable

# Build the state-transition graph
def build_graph(initial_modal_shares, final_states):
    root = StateNode(time=0, modal_shares=initial_modal_shares)
    queue = deque([root])
    paths = []

    while queue:
        current_node = queue.popleft()

        if current_node.time == len(TIME_PERIODS):
            # Check if final biking share matches desired final states
            if np.isclose(current_node.modal_shares['s'], final_states, atol=0.005).any():
                # Backtrack to get the path
                path = []
                node = current_node
                while node.parent is not None:
                    path.append({
                        'time': node.time,
                        'decisions': node.decisions,
                        'modal_shares': node.modal_shares
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

            # Simulate state transition
            modal_shares_next, actions_to_disable = state_transition(
                current_node.modal_shares,
                decisions
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
                disabled_actions=disabled_actions_updated
            )
            current_node.children.append(child_node)
            queue.append(child_node)
    return paths

# Execute the graph building
paths = build_graph(initial_modal_shares, final_states)

# Display the results
print(f"Total feasible paths to desired final states: {len(paths)}\n")

# Find and visualize the most parsimonious path
def count_positive_decisions(path):
    return sum(sum(step['decisions']) for step in path)

def count_repeat_decisions(path):
    decision_sets = [set(i for i, d in enumerate(step['decisions']) if d == 1) for step in path]
    return sum(len(set1.intersection(set2)) for set1, set2 in zip(decision_sets, decision_sets[1:]))

def find_most_parsimonious_path(paths):
    if not paths:
        return None
    
    sorted_paths = sorted(
        paths,
        key=lambda p: (count_positive_decisions(p), count_repeat_decisions(p))
    )
    return sorted_paths[0]

def visualize_parsimonious_path(path):
    times = [0] + [step['time'] for step in path]
    shares = [initial_modal_shares['s']] + [step['modal_shares']['s'] for step in path]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(times, shares, marker='o')
    
    for i, step in enumerate(path):
        decision_names = [DECISIONS[i] for i, d in enumerate(step['decisions']) if d == 1]
        if decision_names:
            ax.annotate(', '.join(decision_names), (step['time'], step['modal_shares']['s']), 
                        textcoords="offset points", xytext=(0,10), ha='center')
    
    ax.set_xlabel('Time Period')
    ax.set_ylabel('Biking Modal Share')
    ax.set_title('Most Parsimonious Path for Increasing Biking Modal Share')
    
    # Add legend
    legend_text = '\n'.join([f"{abbr}: {name}" for abbr, name in zip(DECISIONS, [
        "Improve Public Communication",
        "Anti-Car Propaganda",
        "Raise Prices Public Transport",
        "Invest in Bike Infrastructure",
        "Leverage Bike-Friendly Policies",
        "Levy Tax on Parking"
    ])])
    ax.text(1.05, 0.5, legend_text, transform=ax.transAxes, 
            bbox=dict(facecolor='white', edgecolor='black', alpha=0.7),
            verticalalignment='center')
    
    plt.tight_layout()
    plt.show()

most_parsimonious_path = find_most_parsimonious_path(paths)

if most_parsimonious_path:
    print("Most Parsimonious Path:")
    for step in most_parsimonious_path:
        decision_names = [DECISIONS[i] for i, d in enumerate(step['decisions']) if d == 1]
        modal_shares_str = ', '.join([f"{k.upper()}: {v:.3f}" for k, v in step['modal_shares'].items()])
        print(f"  Time {step['time']}: Decisions: {decision_names}, {modal_shares_str}")
    print("\n")
    
    visualize_parsimonious_path(most_parsimonious_path)
else:
    print("No feasible paths found.")
