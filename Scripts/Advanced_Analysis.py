import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from Data_Analysis_Tools import (load_data, create_generic_columns, perform_chi_square_test,
                                 perform_kruskal_wallis_test, plot_stacked_bar, plot_grouped_bar,
                                 debug_column_values, get_color_palette, parse_multi_select_by_keyword)
import numpy as np
import os

# Create a directory to save plots if it doesn't exist
os.makedirs('plots', exist_ok=True)

def save_plot(fig, filename):
    fig.savefig(os.path.join('plots', filename), bbox_inches='tight', dpi=300)

def categorize_cyclists(df):
    print("\nDebugging Q18 and Q19 before categorization:")
    debug_column_values(df, 'Q18')
    debug_column_values(df, 'Q19')
    
    df['Bike_User'] = ((df['Q18'] == 'Yes') | (df['Q19'] == 'Yes')).astype(int)
    
    print("\nDebugging Bike_User after categorization:")
    debug_column_values(df, 'Bike_User')
    
    print("\nCross-tabulation of Q18, Q19, and Bike_User:")
    print(pd.crosstab([df['Q18'], df['Q19']], df['Bike_User']))
    
    return df

def plot_factor_comparison(data1, data2=None, title="", label1="Bike Users", label2="Non-Bike Users", filename="factor_comparison.png"):
    plt.figure(figsize=(12, 6))
    x = range(len(data1))
    
    if data2 is not None and not data2.empty:
        width = 0.35
        plt.bar([i - width/2 for i in x], data1.values, width, label=label1, color='#4CAF50')  # Solid green for Bike Users
        plt.bar([i + width/2 for i in x], data2.values, width, label=label2, color='#F44336', hatch='///')  # Hatched red for Non-Bike Users
    else:
        plt.bar(x, data1.values, label=label1, color='#4CAF50')  # Solid green for Bike Users
    
    plt.xlabel('Factors')
    plt.ylabel('Percentage')
    plt.title(title)
    plt.xticks(x, data1.index, rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    save_plot(plt.gcf(), filename)
    plt.show()

def analyze_cycling_frequency(df):
    demographic_vars = ['Q3', 'Q4', 'Q5']
    cycling_frequency_order = ['A couple times a year', 'Monthly', 'Weekly', 'Daily']
    
    bike_users = df[df['Bike_User'] == 1]
    
    for var in demographic_vars:
        debug_column_values(bike_users, var)
        
        # Filter out low-frequency categories
        var_counts = bike_users[var].value_counts()
        min_sample_size = max(len(bike_users) * 0.05, 30)
        valid_categories = var_counts[var_counts >= min_sample_size].index
        filtered_bike_users = bike_users[bike_users[var].isin(valid_categories)]
        
        # Filter out "Never" from Q20
        filtered_bike_users = filtered_bike_users[filtered_bike_users['Q20'] != 'Never']
        
        chi2, p_value = perform_chi_square_test(filtered_bike_users, var, 'Q20')
        print(f"Chi-square test results for {var} vs Q20 (Bike Users only, filtered):")
        print(f"Chi-square statistic: {chi2:.2f}, p-value: {p_value:.4f}")
        
        var_description = {
            'Q3': 'Age',
            'Q4': 'Gender',
            'Q5': 'Employment Status'
        }
        
        ax = plot_stacked_bar(filtered_bike_users, var, 'Q20', 
                              f"How often do you cycle? (by {var_description[var]}, Bike Users only)", 
                              var_description[var], "Proportion",
                              order=cycling_frequency_order)
        save_plot(plt.gcf(), f"cycling_frequency_{var}.png")
        plt.show()
        
        print(f"\nExcluded categories for {var_description[var]} due to small sample size (threshold: {min_sample_size}):")
        print(var_counts[var_counts < min_sample_size])

def analyze_cycling_factors(df):
    bike_users = df[df['Bike_User'] == 1]
    non_bike_users = df[df['Bike_User'] == 0]
    
    debug_column_values(bike_users, 'Q21')
    debug_column_values(df, 'Q22')
    
    encouragement_data = bike_users['Q21'].str.get_dummies(sep=',').mean().sort_values(ascending=False)
    
    # Use keyword method for discouragement factors
    discouragement_data_bike_users, _ = parse_multi_select_by_keyword(bike_users, 'Q22')
    discouragement_data_non_bike_users, _ = parse_multi_select_by_keyword(non_bike_users, 'Q22')
    
    discouragement_data_bike_users = pd.Series(discouragement_data_bike_users) / len(bike_users) * 100
    discouragement_data_non_bike_users = pd.Series(discouragement_data_non_bike_users) / len(non_bike_users) * 100
    
    discouragement_data_bike_users = discouragement_data_bike_users.sort_values(ascending=False)
    discouragement_data_non_bike_users = discouragement_data_non_bike_users.reindex(discouragement_data_bike_users.index)
    
    plot_factor_comparison(encouragement_data, title="What encourages you to cycle?", label1="Bike Users", filename="encouragement_factors.png")
    plot_factor_comparison(discouragement_data_bike_users, discouragement_data_non_bike_users, 
                           "What discourages you from cycling?", 
                           "Bike Users", "Non-Bike Users", filename="discouragement_factors.png")

def analyze_commute_distance_vs_cycling(df):
    bike_users = df[df['Bike_User'] == 1]
    debug_column_values(bike_users, 'commute_distance')
    debug_column_values(bike_users, 'Q20')
    
    # Set minimum sample size threshold (e.g., 5% of total samples or 30, whichever is larger)
    min_sample_size = max(len(bike_users) * 0.05, 30)
    
    # Filter out categories with sample sizes below the threshold
    commute_distance_counts = bike_users['commute_distance'].value_counts()
    valid_distances = commute_distance_counts[commute_distance_counts >= min_sample_size].index
    
    filtered_bike_users = bike_users[bike_users['commute_distance'].isin(valid_distances)]
    filtered_bike_users = filtered_bike_users[filtered_bike_users['Q20'] != 'Never']
    
    h_statistic, p_value = perform_kruskal_wallis_test(filtered_bike_users, 'commute_distance', 'Q20')
    print(f"Kruskal-Wallis test results for commute_distance vs Q20 (Bike Users, filtered):")
    print(f"H-statistic: {h_statistic:.2f}, p-value: {p_value:.4f}")
    
    # Create a custom color palette
    cycling_frequency_order = ['A couple times a year', 'Monthly', 'Weekly', 'Daily']
    color_palette = dict(zip(cycling_frequency_order, plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(cycling_frequency_order)))))
    
    # Define the correct order for commute distances
    distance_order = ['<2 km', '2-5 km', '5-10 km', '10-20 km', '20-50 km', '>50 km']
    
    # Filter the distance_order to include only valid distances
    valid_distance_order = [d for d in distance_order if d in valid_distances]
    
    plt.figure(figsize=(12, 6))
    ax = sns.countplot(x='commute_distance', hue='Q20', data=filtered_bike_users, 
                       hue_order=cycling_frequency_order, palette=color_palette,
                       order=valid_distance_order)  # Specify the order here
    
    plt.title("Cycling Frequency by Commute Distance (Bike Users)")
    plt.xlabel("Commute Distance")
    plt.ylabel("Count")
    plt.legend(title="Cycling Frequency", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    save_plot(plt.gcf(), "commute_distance_vs_cycling.png")
    plt.show()
    
    print(f"\nExcluded categories due to small sample size (threshold: {min_sample_size}):")
    print(commute_distance_counts[commute_distance_counts < min_sample_size])

def analyze_infrastructure_perception(df):
    debug_column_values(df, 'Q30')
    chi2, p_value = perform_chi_square_test(df, 'Q30', 'Bike_User')
    print(f"Chi-square test results for Q30 vs Bike_User:")
    print(f"Chi-square statistic: {chi2:.2f}, p-value: {p_value:.4f}")
    
    ax = plot_stacked_bar(df, 'Q30', 'Bike_User', 
                          "Infrastructure Perception by Bike User", "Infrastructure Perception", "Proportion")
    save_plot(plt.gcf(), "infrastructure_perception.png")
    plt.show()

def analyze_policy_support(df):
    debug_column_values(df, 'Q33')
    policy_support = df['Q33'].str.get_dummies(sep=',').mean().sort_values(ascending=False)
    
    plt.figure(figsize=(12, 6))
    colors = plt.cm.RdYlGn(np.linspace(0.8, 0.2, len(policy_support)))  # Reverse the color order
    policy_support.plot(kind='bar', color=colors)
    plt.title("Which policies would encourage more cycling in Prague?")
    plt.xlabel("Policy")
    plt.ylabel("Proportion of Support")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    save_plot(plt.gcf(), "policy_support.png")
    plt.show()
    
    # Create dummy variables for each policy
    policy_dummies = df['Q33'].str.get_dummies(sep=',')
    df = pd.concat([df, policy_dummies], axis=1)
    
    for policy in policy_support.index:
        chi2, p_value = perform_chi_square_test(df, policy, 'Bike_User')
        print(f"Chi-square test results for {policy} vs Bike_User:")
        print(f"Chi-square statistic: {chi2:.2f}, p-value: {p_value:.4f}")

def main():
    file_path = r"V:\Prague_Biking\Data\Survey Data\Excel survey results full version (1).xlsx"
    df = load_data(file_path, skiprows=[1])
    
    print(f"NaN values in Q20 before creating generic columns: {df['Q20'].isna().sum()}")
    
    column_mappings = {
        'commute_modality': ('Q17', 'Q14'),
        'commute_distance': ('Q15', 'Q12'),
        'commute_time': ('Q16', 'Q13')
    }
    df = create_generic_columns(df, column_mappings)
    
    print(f"NaN values in Q20 after creating generic columns: {df['Q20'].isna().sum()}")
    
    df = categorize_cyclists(df)
    
    analyze_cycling_frequency(df)
    analyze_cycling_factors(df)
    analyze_commute_distance_vs_cycling(df)
    analyze_infrastructure_perception(df)
    analyze_policy_support(df)

if __name__ == "__main__":
    main()
