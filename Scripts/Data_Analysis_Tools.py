import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency, kruskal
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors

def cramers_v(confusion_matrix):
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    min_dim = min(confusion_matrix.shape) - 1
    return np.sqrt(chi2 / (n * min_dim))

def analyze_distributions(df, columns, tee=True):
    results = {}
    for col_info in columns:
        col, title = col_info[:2]
        multiple_choice = col_info[2] if len(col_info) > 2 else False
        valid_data = df.dropna(subset=[col])
        distribution = valid_data[col].str.split(',', expand=True).stack().value_counts(normalize=True) if multiple_choice else valid_data[col].value_counts(normalize=True)
        distribution = distribution[distribution.index != 'Unknown']
        
        if tee:
            print(f"\n{title}")
            for option, proportion in distribution.items():
                if option:
                    print(f"{option}: {proportion:.2%}")
        
        results[col] = distribution
    return results

def statistical_analysis(contingency_table, min_sample_size=30):
    chi2, p_value, dof, expected = chi2_contingency(contingency_table)
    reliable = not ((expected < 5).any() or (contingency_table.sum(axis=1) < min_sample_size).any())
    effect_size = cramers_v(contingency_table) if p_value < 0.05 and reliable else None
    
    return {
        'chi2': chi2, 'p_value': p_value, 'dof': dof, 'reliable': reliable,
        'reason': "Low expected frequencies or small sample size." if not reliable else "",
        'effect_size': effect_size, 'total_samples': contingency_table.sum().sum()
    }

def analyze_cross_tabulations(df, cross_tabs, tee=True):
    results = {}
    for tab in cross_tabs:
        col1, col2, title = tab[:3]
        multiple_choice = tab[3] if len(tab) > 3 else False
        valid_df = df.dropna(subset=[col1, col2]).copy()  # Create an explicit copy
        
        if multiple_choice:
            all_modalities = set(','.join(valid_df[col2].dropna()).split(','))
            for modality in all_modalities:
                valid_df.loc[:, f'{modality}_prop'] = valid_df[col2].apply(lambda x: 1 / len(x.split(',')) if modality in x else 0)
            crosstab = valid_df.groupby(col1)[[f'{m}_prop' for m in all_modalities]].mean()
            crosstab.columns = [col.replace('_prop', '') for col in crosstab.columns]
        else:
            crosstab = pd.crosstab(valid_df[col1], valid_df[col2], normalize='index')
        
        if tee:
            print(f"\n{title}")
            print(f"Number of valid rows: {len(valid_df)}")
            print("\nCrosstab:")
            print(crosstab.round(2))
            
            if not multiple_choice:
                stats = statistical_analysis(pd.crosstab(valid_df[col1], valid_df[col2]))
                print("\nStatistical analysis:")
                print(f"Chi-square: {stats['chi2']:.2f}, p-value: {stats['p_value']:.4f}")
                if not stats['reliable']:
                    print(f"Warning: Results may be unreliable. Reason: {stats['reason']}")
                if stats['effect_size']:
                    print(f"Effect size (Cramer's V): {stats['effect_size']:.4f}")
        
        results[title] = crosstab
    return results

def analyze_likert_scales(df, likert_scales, tee=True):
    results = {}
    for scale, title in likert_scales:
        likert_distribution = df[scale].value_counts(normalize=True).sort_index()
        
        if tee:
            print(f"\n{title}")
            for option, proportion in likert_distribution.items():
                print(f"{option}: {proportion:.2%}")
        
        results[scale] = likert_distribution
    return results

def load_data(file_path, skiprows=None):
    return pd.read_excel(file_path, skiprows=skiprows)

def create_generic_columns(df, column_mappings):
    for new_col, (col1, col2) in column_mappings.items():
        df[new_col] = df[col1].fillna(df[col2])
    return df

def print_nan_info(df, columns, tee=True):
    nan_info = {col: (count := df[col].isna().sum(), count / len(df) * 100) for col in columns}
    if tee:
        for col, (count, percentage) in nan_info.items():
            print(f"{col}: {count} NaN values ({percentage:.2f}%)")
    return nan_info

def perform_chi_square_test(df, col1, col2):
    contingency_table = pd.crosstab(df[col1], df[col2])
    chi2, p_value, dof, expected = chi2_contingency(contingency_table)
    return chi2, p_value

def perform_kruskal_wallis_test(df, group_col, value_col):
    groups = [group for _, group in df.groupby(group_col)[value_col]]
    h_statistic, p_value = kruskal(*groups)
    return h_statistic, p_value

def create_color_palette(n_colors):
    green = mcolors.to_rgb('#4CAF50')  # Muted green
    red = mcolors.to_rgb('#F44336')  # Muted red
    return [mcolors.to_hex(mcolors.rgb_to_hsv(
        tuple(red[j] + (i/(n_colors-1))*(green[j]-red[j]) for j in range(3))
    )) for i in range(n_colors)]

def plot_stacked_bar(df, x, y, title, xlabel, ylabel, order=None):
    df_plot = df.groupby(x)[y].value_counts(normalize=True).unstack()
    if order:
        df_plot = df_plot.reindex(columns=order)
    
    ax = df_plot.plot(kind='bar', stacked=True, figsize=(12, 6), 
                      colormap='RdYlGn')
    
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(title=y, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    return ax

def plot_grouped_bar(df, x, y, hue, title, xlabel, ylabel):
    plt.figure(figsize=(12, 6))
    # Use the 'RdYlGn' colormap, which goes from red to yellow to green
    ax = sns.barplot(x=x, y=y, hue=hue, data=df, palette='RdYlGn')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(title=hue, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    return ax

# Add this function at the end of the file

def debug_column_values(df, column, n=10):
    print(f"\nDebugging column: {column}")
    print(f"Unique values (top {n}):")
    print(df[column].value_counts().head(n))
    print(f"\nNull values: {df[column].isnull().sum()}")
    print(f"NaN values: {df[column].isna().sum()}")
    print(f"Empty strings: {(df[column] == '').sum()}")

def get_color_palette(n_colors):
    return plt.cm.RdYlGn(np.linspace(0, 1, n_colors))
def parse_multi_select_by_keyword(df, column):
    keywords = {
        'Q28': {
            "Physical health": "Physical health improvements",
            "Mental health": "Mental health improvements",
            "environment": "Better for the environment (air quality, urban biodiversity, etc.)",
            "Faster": "Faster or more efficient transportation",
            "less busy": "Makes the city less busy / traffic noise reduction",
            "Economic": "Economic benefits",
            "does not bring any benefits": "Cycling does not bring any benefits"
        },
        'Q29': {
            "Safety concerns": "Safety concerns",
            "Mental health problems": "Mental health problems",
            "Physical health problems": "Physical health problems (respiratory sickness, etc.)",
            "Traffic disruptions": "Traffic disruptions",
            "More conflict": "More conflict with pedestrian / car users",
            "does not bring any disbenefits": "Cycling does not bring any disbenefits"
        },
        'Q22': {
            "distance is too far": "The distance is too far",
            "transport (e.g. car) is much more convenient": "Public transport / private transport (e.g. car) is much more convenient",
            "Poor cycling facilities": "Poor cycling facilities (e.g., bike lanes, bike parking)",
            "Safety issues": "Safety issues",
            "Physical health reasons": "Physical health reasons",
            "stereotypical image": "The stereotypical image of cyclists",
            "Weather conditions": "Weather conditions",
            "Nothing discourages": "Nothing discourages me from cycling",
            "Difficult Terrain": "Difficult Terrain",
            "do not own a bicycle": "I do not own a bicycle",
            "Mental health reasons": "Mental health reasons"
        },
        'Q21': {
            "Physical health reasons": "Physical health reasons / fitness",
            "Mental health reasons": "Mental health reasons",
            "Time efficiency": "Time efficiency / independence from public transport schedule",
            "Cost-saving reasons": "Cost-saving reasons",
            "Environmental concerns": "Environmental concerns",
            "Enjoyment purpose": "Enjoyment purpose",
            "Being outside in nature": "Being outside in nature",
            "Friends / family cycle": "Friends / family cycle too",
            "Nothing encourages": "Nothing encourages me to cycle"
        },
        'Q33': {
            "Improved bike lanes": "Improved bike lanes and infrastructure",
            "Bike-sharing": "Expanded bike-sharing programs",
            "Integration with public transport": "Better integration of cycling with public transport",
            "Financial incentives": "Financial incentives for cycling",
            "Education and awareness": "Education and awareness programs",
            "Traffic calming": "Traffic calming measures",
            "Bike parking": "Improved bike parking facilities",
            "No support": "No support for any cycling policies"
        }
    }

    options = df[column].dropna()
    matched_options = {}
    unmatched_options = set()

    for response in options:
        matched = False
        for key, full_text in keywords.get(column, {}).items():
            if key.lower() in response.lower():
                matched_options[full_text] = matched_options.get(full_text, 0) + 1
                matched = True
        if not matched and "Other" not in response:
            unmatched_options.add(response)

    return matched_options, unmatched_options