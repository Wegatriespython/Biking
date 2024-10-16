import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from Data_Analysis_Tools import (load_data, create_generic_columns, perform_chi_square_test,
                                 debug_column_values, get_color_palette)
import numpy as np
import os
from print_unique_values import parse_multi_select_by_keyword

# Create a directory to save plots if it doesn't exist
os.makedirs('plots', exist_ok=True)

def save_plot(fig, filename):
    fig.savefig(os.path.join('plots', filename), bbox_inches='tight', dpi=300)

def plot_factor_comparison(data1, data2, title, label1, label2, filename):
    plt.figure(figsize=(12, 6))
    x = range(len(data1))
    width = 0.35
    
    plt.bar([i - width/2 for i in x], data1.values, width, label=label1, color='#4CAF50')
    plt.bar([i + width/2 for i in x], data2.values, width, label=label2, color='#F44336', hatch='///')
    
    plt.xlabel('Factors')
    plt.ylabel('Percentage')
    plt.title(title)
    plt.xticks(x, data1.index, rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    save_plot(plt.gcf(), filename)
    plt.show()

def analyze_gender_effect(df, column, title, filename):
    bike_users = df[df['Bike_User'] == 1]
    male_data = bike_users[bike_users['Q4'] == 'Male']
    female_data = bike_users[bike_users['Q4'] == 'Female']
    
    male_counts, _ = parse_multi_select_by_keyword(male_data, column)
    female_counts, _ = parse_multi_select_by_keyword(female_data, column)
    
    male_percentages = pd.Series(male_counts) / len(male_data) * 100
    female_percentages = pd.Series(female_counts) / len(female_data) * 100
    
    plot_factor_comparison(male_percentages, female_percentages, title, "Male", "Female", filename)

def analyze_by_category(df, column, category_col, title_prefix, filename_prefix, categories_to_show=None):
    if categories_to_show is None:
        categories = df[category_col].unique()
    else:
        categories = categories_to_show
    
    category_data = {}
    for category in categories:
        category_df = df[df[category_col] == category]
        if column == 'Q33':
            # Use original CSV parsing method for Q33
            data = category_df[column].str.get_dummies(sep=',').mean()
        else:
            # Use new keyword-based method for other questions
            counts, _ = parse_multi_select_by_keyword(category_df, column)
            data = pd.Series(counts) / len(category_df) * 100
        category_data[category] = data
    
    all_factors = set()
    for data in category_data.values():
        all_factors.update(data.index)
    
    plt.figure(figsize=(12, 6))
    x = range(len(all_factors))
    width = 0.8 / len(categories)
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(categories)))
    
    for i, (category, data) in enumerate(category_data.items()):
        values = [data.get(factor, 0) for factor in all_factors]
        plt.bar([xi + i*width for xi in x], values, width, label=category, color=colors[i])
    
    plt.title(f"{title_prefix}")
    plt.xlabel("Factors")
    plt.ylabel("Percentage")
    plt.xticks([xi + width*(len(categories)-1)/2 for xi in x], all_factors, rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    save_plot(plt.gcf(), f"{filename_prefix}.png")
    plt.show()

def analyze_goldilocks_effect(df, column, title, filename):
    df['Goldilocks'] = df['commute_distance'].apply(lambda x: 'Below 5km' if x in ['<2 km', '2-5 km'] else 'Above 5km')
    below_5km = df[df['Goldilocks'] == 'Below 5km']
    above_5km = df[df['Goldilocks'] == 'Above 5km']
    
    below_5km_counts, _ = parse_multi_select_by_keyword(below_5km, column)
    above_5km_counts, _ = parse_multi_select_by_keyword(above_5km, column)
    
    below_5km_percentages = pd.Series(below_5km_counts) / len(below_5km) * 100
    above_5km_percentages = pd.Series(above_5km_counts) / len(above_5km) * 100
    
    plot_factor_comparison(below_5km_percentages, above_5km_percentages, title, "Below 5km", "Above 5km", filename)

def main():
    file_path = r"V:\Prague_Biking\Data\Survey Data\Excel survey results full version (1).xlsx"
    df = load_data(file_path, skiprows=[1])
    
    column_mappings = {
        'commute_modality': ('Q17', 'Q14'),
        'commute_distance': ('Q15', 'Q12'),
        'commute_time': ('Q16', 'Q13')
    }
    df = create_generic_columns(df, column_mappings)
    
    df['Bike_User'] = ((df['Q18'] == 'Yes') | (df['Q19'] == 'Yes')).astype(int)
    
    # Gender effect on encouragement and discouragement
    analyze_gender_effect(df, 'Q21', "Encouragement Factors by Gender (Bike Users)", "encouragement_by_gender.png")
    analyze_gender_effect(df, 'Q22', "Discouragement Factors by Gender (Bike Users)", "discouragement_by_gender.png")
    
    # Benefits and disbenefits by age and employment
    age_groups = ['18 - 24', '25 - 34', '35 - 44']
    analyze_by_category(df, 'Q28', 'Q3', "Benefits of Cycling by Age Group", "benefits_by_age", categories_to_show=age_groups)
    analyze_by_category(df, 'Q29', 'Q3', "Disbenefits of Cycling by Age Group", "disbenefits_by_age", categories_to_show=age_groups)
    
    employment_groups = ['Student', 'Employed']
    analyze_by_category(df, 'Q28', 'Q5', "Benefits of Cycling by Employment Status", "benefits_by_employment", categories_to_show=employment_groups)
    analyze_by_category(df, 'Q29', 'Q5', "Disbenefits of Cycling by Employment Status", "disbenefits_by_employment", categories_to_show=employment_groups)
    
    # Policy support by gender and employment
    analyze_by_category(df, 'Q33', 'Q4', "Policy Support by Gender", "policy_support_by_gender", categories_to_show=['Male', 'Female'])
    analyze_by_category(df, 'Q33', 'Q5', "Policy Support by Employment Status", "policy_support_by_employment", categories_to_show=employment_groups)
    
    # Goldilocks effect on encouragement and discouragement
    analyze_goldilocks_effect(df, 'Q21', "Encouragement Factors by Commute Distance", "encouragement_by_distance.png")
    analyze_goldilocks_effect(df, 'Q22', "Discouragement Factors by Commute Distance", "discouragement_by_distance.png")

if __name__ == "__main__":
    main()
