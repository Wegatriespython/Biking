import itertools
import numpy as np
import matplotlib.pyplot as plt

# Define constants and parameters
TIME_PERIODS = [1, 2, 3]
DECISIONS = ['IPC', 'ACP', 'RPPT', 'IBI', 'LBF', 'LTP']
DECISION_INDICES = {name: idx for idx, name in enumerate(DECISIONS)}

# Effect sizes (Assumed values for illustration; these need to be defined based on data)
delta = {
    'ACP': 0.01,   # Effect of Anti-Car Propaganda
    'RPPT': 0.005, # Effect of Raise Prices Public Transport
    'IBI': 0.02,   # Effect of Invest in Bike Infrastructure
    'LTP': 0.007   # Effect of Levy Tax on Parking
}

# Initial and final states
s0 = 0.01  # Initial biking modal share (Assumed value)
final_states = [0.20, 0.15, 0.10]  # Good, Medium, Baseline scenarios

# Possible decision combinations at each time period (64 combinations)
decision_space = list(itertools.product([0, 1], repeat=6))

# State node representation
class StateNode:
    def __init__(self, time, biking_share, decisions=None, parent=None):
        self.time = time
        self.biking_share = biking_share
        self.decisions = decisions  # Decisions taken to reach this state
        self.parent = parent        # Parent StateNode
        self.children = []          # Child StateNodes

    def __repr__(self):
        return f"Time: {self.time}, Share: {self.biking_share:.3f}, Decisions: {self.decisions}"

# Function to compute the probability of IBI success
def p_IBI(D_IPC, D_LBF):
    if D_IPC == 1 and D_LBF == 1:
        return 1.0
    elif D_IPC == 0 and D_LBF == 0:
        return 0.2
    else:
        return 0.6  # Intermediate probability (assumed value)

# State transition function
def state_transition(s_prev, decisions):
    s = s_prev
    D = decisions

    # Unpack decision variables
    D_IPC, D_ACP, D_RPPT, D_IBI, D_LBF, D_LTP = D

    # Effect of ACP
    s += D_ACP * delta['ACP']

    # Effect of RPPT
    s += D_RPPT * delta['RPPT']

    # Probability of IBI success
    p_ibi = p_IBI(D_IPC, D_LBF)
    IBI_success = D_IBI * p_ibi

    # Effect of IBI
    s += IBI_success * delta['IBI']

    # Effect of LTP
    s += D_LTP * delta['LTP']

    # Ensure biking share is within bounds [0, 1]
    s = min(max(s, 0), 1)
    return s

# Build the state-transition graph
from collections import deque

def build_graph(s0, final_states):
    root = StateNode(time=0, biking_share=s0)
    queue = deque([root])
    paths = []

    while queue:
        current_node = queue.popleft()

        if current_node.time == len(TIME_PERIODS):
            # Check if final state matches desired final states
            if np.isclose(current_node.biking_share, final_states, atol=0.005).any():
                # Backtrack to get the path
                path = []
                node = current_node
                while node.parent is not None:
                    path.append((node.time, node.decisions, node.biking_share))
                    node = node.parent
                path.reverse()
                paths.append(path)
            continue

        # Generate possible decisions at this time period
        for decisions in decision_space:
            # Apply logical constraints if any (e.g., resource limitations)
            # For this example, we proceed with all combinations

            # Compute next state
            s_next = state_transition(current_node.biking_share, decisions)

            # Create child node
            child_node = StateNode(
                time=current_node.time + 1,
                biking_share=s_next,
                decisions=decisions,
                parent=current_node
            )

            current_node.children.append(child_node)
            queue.append(child_node)

    return paths

# Execute the graph building
paths = build_graph(s0, final_states)

# Display the results
print(f"Total feasible paths to desired final states: {len(paths)}\n")

"""for idx, path in enumerate(paths):
    print(f"Path {idx + 1}:")
    for time, decisions, biking_share in path:
        decision_names = [DECISIONS[i] for i, d in enumerate(decisions) if d == 1]
        print(f"  Time {time}: Decisions: {decision_names}, Biking Share: {biking_share:.3f}")
    print("\n")"""

def count_positive_decisions(path):
    return sum(sum(decisions) for _, decisions, _ in path)

def count_repeat_decisions(path):
    decision_sets = [set(i for i, d in enumerate(decisions) if d == 1) for _, decisions, _ in path]
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
    times = [0] + [t for t, _, _ in path]
    shares = [s0] + [share for _, _, share in path]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(times, shares, marker='o')
    
    for i, (time, decisions, share) in enumerate(path):
        decision_names = [DECISIONS[i] for i, d in enumerate(decisions) if d == 1]
        if decision_names:
            ax.annotate(', '.join(decision_names), (time, share), 
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

# After building the graph and finding paths
paths = build_graph(s0, final_states)

print(f"Total feasible paths to desired final states: {len(paths)}\n")

# Find and visualize the most parsimonious path
most_parsimonious_path = find_most_parsimonious_path(paths)

if most_parsimonious_path:
    print("Most Parsimonious Path:")
    for time, decisions, biking_share in most_parsimonious_path:
        decision_names = [DECISIONS[i] for i, d in enumerate(decisions) if d == 1]
        print(f"  Time {time}: Decisions: {decision_names}, Biking Share: {biking_share:.3f}")
    print("\n")
    
    visualize_parsimonious_path(most_parsimonious_path)
else:
    print("No feasible paths found.")
