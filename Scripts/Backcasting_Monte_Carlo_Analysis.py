import csv
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

def read_csv(filename):
    return pd.read_csv(filename)

def analyze_decisions_by_time(data):
    decisions_by_time = defaultdict(lambda: defaultdict(int))
    for _, row in data.iterrows():
        time = int(row['Time'])
        decisions = row['Decisions']
        if isinstance(decisions, str):
            for decision in decisions.split('|'):
                if decision:
                    decisions_by_time[time][decision] += 1
        else:
            decisions_by_time[time]['No Action'] += 1
    return decisions_by_time

def plot_most_common_decisions(decisions_by_time):
    num_times = len(decisions_by_time)
    fig, axs = plt.subplots(1, num_times, figsize=(5*num_times, 5), sharey=True)
    
    legend_text = {
        "IPC": "Improve Policy Coordination",
        "ACP": "Anti-Car Propaganda",
        "RPPT": "Raise Prices Public Transport",
        "IBI": "Invest in Bike Infrastructure",
        "LBF": "Lobby for Biking Finance",
        "LTP": "Levy Tax on Car Parking"
    }
    
    for time, ax in zip(sorted(decisions_by_time.keys()), axs):
        decisions = decisions_by_time[time]
        sorted_decisions = sorted(decisions.items(), key=lambda x: x[1], reverse=True)
        
        decisions, counts = zip(*sorted_decisions[:5])  # Top 5 decisions
        bars = ax.bar(decisions, counts)
        ax.set_title(f"Time {time}")
        ax.set_xlabel("Decisions")
        ax.tick_params(axis='x', rotation=45)
        
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height}', ha='center', va='bottom')
    
    axs[0].set_ylabel("Frequency")
    plt.tight_layout()
    
    # Add legend
    fig.subplots_adjust(bottom=0.3)
    plt.figtext(0.5, 0.02, "\n".join([f"{k}: {v}" for k, v in legend_text.items()]),
                ha="center", fontsize=10, bbox={"facecolor":"orange", "alpha":0.5, "pad":5})
    
    plt.savefig("most_common_decisions.png")
    plt.close()

def plot_modal_shares_boxplot(data):
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=data[['Modal_Shares_s', 'Modal_Shares_c', 'Modal_Shares_p', 'Modal_Shares_o']])
    plt.title("Distribution of Modal Shares")
    plt.ylabel("Share")
    plt.xticks([0, 1, 2, 3], ['Bike', 'Car', 'Public Transport', 'Other'])
    plt.savefig("modal_shares_boxplot.png")
    plt.close()

def plot_bike_share_distribution(data):
    time_4_bike_shares = data[data['Time'] == 4]['Modal_Shares_s']
    
    plt.figure(figsize=(12, 6))
    sns.histplot(time_4_bike_shares, bins=50, kde=True)
    plt.title("Distribution of Time 4 Bike Share")
    plt.xlabel("Bike Share Value")
    plt.ylabel("Frequency")
    plt.savefig("time_4_bike_share_distribution.png")
    plt.close()

def plot_cost_distribution(data):
    plt.figure(figsize=(12, 6))
    sns.histplot(data['Cost'], bins=50, kde=True)
    plt.title("Distribution of Total Costs")
    plt.xlabel("Cost")
    plt.ylabel("Frequency")
    plt.savefig("cost_distribution.png")
    plt.close()

def plot_modal_share_evolution(data):
    avg_shares = data.groupby('Time')[['Modal_Shares_s', 'Modal_Shares_c', 'Modal_Shares_p', 'Modal_Shares_o']].mean()
    
    plt.figure(figsize=(12, 6))
    for mode, name in zip(['Modal_Shares_s', 'Modal_Shares_c', 'Modal_Shares_p', 'Modal_Shares_o'], 
                          ['Bike', 'Car', 'Public Transport', 'Other']):
        plt.plot(avg_shares.index, avg_shares[mode], marker='o', label=name)
    
    plt.title("Average Modal Share Evolution")
    plt.xlabel("Time")
    plt.ylabel("Average Share")
    plt.legend()
    plt.grid(True)
    plt.savefig("modal_share_evolution.png")
    plt.close()

def plot_decision_heatmap(data):
    decision_matrix = pd.DataFrame(0, index=range(1, 5), columns=DECISIONS + ['No Action'])
    for _, row in data.iterrows():
        time = int(row['Time'])
        decisions = row['Decisions']
        if isinstance(decisions, str):
            for decision in decisions.split('|'):
                if decision:
                    decision_matrix.loc[time, decision] += 1
        else:
            decision_matrix.loc[time, 'No Action'] += 1
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(decision_matrix, annot=True, fmt='d', cmap='YlOrRd')
    plt.title("Decision Frequency Heatmap")
    plt.xlabel("Decisions")
    plt.ylabel("Time")
    plt.savefig("decision_heatmap.png")
    plt.close()

# Main execution
DECISIONS = ["IPC", "ACP", "RPPT", "IBI", "LBF", "LTP"]
try:
    data = read_csv("best_paths.csv")
    print(f"Loaded {len(data)} rows of data")
    print(data.dtypes)  # Print data types of columns
    print(data['Decisions'].value_counts(dropna=False).head())  # Print most common values in Decisions column

    decisions_by_time = analyze_decisions_by_time(data)
    plot_most_common_decisions(decisions_by_time)
    plot_modal_shares_boxplot(data)
    plot_bike_share_distribution(data)
    plot_cost_distribution(data)
    plot_modal_share_evolution(data)
    plot_decision_heatmap(data)

    # Print some statistics
    print("Most common decisions by time:")
    for time, decisions in sorted(decisions_by_time.items()):
        print(f"Time {time}:")
        for decision, count in sorted(decisions.items(), key=lambda x: x[1], reverse=True)[:3]:
            print(f"  {decision}: {count}")
        print()

    print("Average modal shares:")
    for mode in ['Modal_Shares_s', 'Modal_Shares_c', 'Modal_Shares_p', 'Modal_Shares_o']:
        print(f"{mode}: {data[mode].mean():.3f}")

    print("\nFinal bike share statistics:")
    final_bike_share = data[data['Time'] == 4]['Modal_Shares_s']
    print(f"Mean: {final_bike_share.mean():.3f}")
    print(f"Median: {final_bike_share.median():.3f}")
    print(f"Std Dev: {final_bike_share.std():.3f}")
    print(f"Min: {final_bike_share.min():.3f}")
    print(f"Max: {final_bike_share.max():.3f}")

    print("\nTotal cost statistics:")
    print(f"Mean: {data['Cost'].mean():.3f}")
    print(f"Median: {data['Cost'].median():.3f}")
    print(f"Std Dev: {data['Cost'].std():.3f}")
    print(f"Min: {data['Cost'].min():.3f}")
    print(f"Max: {data['Cost'].max():.3f}")

except Exception as e:
    print(f"An error occurred: {e}")
    import traceback
    traceback.print_exc()
