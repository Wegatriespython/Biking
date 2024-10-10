import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re

# File paths
categories = ['Online_508', 'Offline_508', 'Online_Rest', 'Offline_Rest']
file_paths = {cat: f"V:\\Prague_Biking\\Data\\Survey Data\\Processed_Cycling_in_Prague_Survey_{cat}.xlsx" for cat in categories}

def convert_distance(distance):
    if pd.isna(distance):
        return None
    if isinstance(distance, str):
        if '<2 km' in distance:
            return 1
        elif '2-5 km' in distance:
            return 3.5
        elif '5-10 km' in distance:
            return 7.5
        elif '10-20 km' in distance:
            return 15
        elif '20-50 km' in distance:
            return 35
        elif '>50 km' in distance:
            return 60
    return distance

def convert_time(time):
    if pd.isna(time):
        return None
    if isinstance(time, str):
        if '<10 minutes' in time:
            return 5
        elif '11-30 minutes' in time:
            return 20
        elif '31-60 minutes' in time:
            return 45
        elif '61-90 minutes' in time:
            return 75
        elif '91-120 minutes' in time:
            return 105
        elif '>120 minutes' in time:
            return 135
    return time

def process_modal_split(modal_split):
    if pd.isna(modal_split):
        return {}
    
    if isinstance(modal_split, (float, int)):
        return {}  # or handle numeric values as needed
    
    transport_counts = {}
    modes = modal_split.split(',')
    for mode in modes:
        clean_mode = mode.strip().lower()
        if 'bus' in clean_mode:
            transport_counts['bus'] = transport_counts.get('bus', 0) + 1
        elif 'car' in clean_mode:
            transport_counts['car'] = transport_counts.get('car', 0) + 1
        elif 'metro' in clean_mode:
            transport_counts['metro'] = transport_counts.get('metro', 0) + 1
        elif 'tram' in clean_mode:
            transport_counts['tram'] = transport_counts.get('tram', 0) + 1
        elif 'cycling' in clean_mode:
            transport_counts['cycling'] = transport_counts.get('cycling', 0) + 1
        elif 'train' in clean_mode:
            transport_counts['train'] = transport_counts.get('train', 0) + 1
        elif 'foot' in clean_mode:
            transport_counts['foot'] = transport_counts.get('foot', 0) + 1
        # Add more conditions for other transport modes as needed
    
    return transport_counts

def preprocess_data(df):
    # Create combined bike user indicator
    df['Bike_User'] = ((df['Q18'] == 'Yes') | (df['Q19'] == 'Yes')).astype(int)
    
    # Convert distance and time to continuous variables
    df['Avg_Distance'] = df['Q15'].apply(convert_distance)
    df['Avg_Time'] = df['Q16'].apply(convert_time)
    
    # Process modal split
    modal_split = df['Q17'].apply(process_modal_split).apply(pd.Series).fillna(0)
    
    # Normalize modal split to get proportions
    modal_split_sum = modal_split.sum(axis=1)
    modal_split = modal_split.div(modal_split_sum, axis=0).fillna(0)
    
    df = pd.concat([df, modal_split], axis=1)
    
    return df

def create_cross_tab_heatmap(df, x, y, title, filename):
    plt.figure(figsize=(12, 8))
    cross_tab = pd.crosstab(df[x], df[y], normalize='index')
    sns.heatmap(cross_tab, annot=True, cmap='YlGnBu', fmt='.2%')
    plt.title(title, fontsize=16)
    plt.tight_layout()
    plt.savefig(f"V:\\Prague_Biking\\Data\\Survey Data\\{filename}.png", dpi=300)
    plt.close()

def create_stacked_bar(df, x, y, title, filename):
    plt.figure(figsize=(12, 8))
    cross_tab = pd.crosstab(df[x], df[y], normalize='index')
    cross_tab.plot(kind='bar', stacked=True)
    plt.title(title, fontsize=16)
    plt.legend(title=y, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f"V:\\Prague_Biking\\Data\\Survey Data\\{filename}.png", dpi=300)
    plt.close()

def analyze_and_visualize(df):
    # Age vs Bike Usage
    create_cross_tab_heatmap(df, 'Q3', 'Bike_User', 'Age vs Bike Usage', 'age_vs_bike_usage')

    # Gender vs Perceived Safety (Q25)
    create_stacked_bar(df, 'Q4', 'Q25', 'Gender vs Perceived Safety', 'gender_vs_safety')

    # Occupation vs Average Commute Distance
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Q5', y='Avg_Distance', data=df)
    plt.title('Occupation vs Average Commute Distance', fontsize=16)
    plt.xlabel('Occupation')
    plt.ylabel('Average Distance (km)')
    plt.tight_layout()
    plt.savefig("V:\\Prague_Biking\\Data\\Survey Data\\occupation_vs_distance.png", dpi=300)
    plt.close()

    # Age vs Average Commute Time
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Q3', y='Avg_Time', data=df)
    plt.title('Age vs Average Commute Time', fontsize=16)
    plt.xlabel('Age Group')
    plt.ylabel('Average Time (minutes)')
    plt.tight_layout()
    plt.savefig("V:\\Prague_Biking\\Data\\Survey Data\\age_vs_commute_time.png", dpi=300)
    plt.close()

    # Modal Split by Age Group
    modal_columns = ['bus', 'car', 'metro', 'tram', 'cycling', 'train', 'foot']
    modal_split = df.groupby('Q3')[modal_columns].mean()
    modal_split.plot(kind='bar', stacked=True, figsize=(12, 6))
    plt.title('Modal Split by Age Group', fontsize=16)
    plt.xlabel('Age Group')
    plt.ylabel('Proportion')
    plt.legend(title='Transport Mode', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig("V:\\Prague_Biking\\Data\\Survey Data\\modal_split_by_age.png", dpi=300)
    plt.close()

    # Modal Split by Gender
    modal_columns = ['bus', 'car', 'metro', 'tram', 'cycling', 'train', 'foot']
    modal_split_by_gender = df.groupby('Q4')[modal_columns].mean()
    
    plt.figure(figsize=(12, 6))
    modal_split_by_gender.plot(kind='bar', stacked=True)
    plt.title('Modal Split by Gender', fontsize=16)
    plt.xlabel('Gender')
    plt.ylabel('Proportion')
    plt.legend(title='Transport Mode', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig("V:\\Prague_Biking\\Data\\Survey Data\\modal_split_by_gender.png", dpi=300)
    plt.close()

# Load and preprocess all data
all_data = []
for category, file_path in file_paths.items():
    df = pd.read_excel(file_path)
    df = preprocess_data(df)
    df['Category'] = category
    all_data.append(df)

combined_df = pd.concat(all_data, ignore_index=True)

# Analyze and visualize the combined dataset
analyze_and_visualize(combined_df)

print("Visualizations complete. Plots have been saved.")

def print_summary_statistics(df):
    modal_columns = ['bus', 'car', 'metro', 'tram', 'cycling', 'train', 'foot']
    print("Modal Split Summary Statistics:")
    print(df[modal_columns].describe())
    
    print("\nCorrelation between modal split and other variables:")
    correlation = df[modal_columns + ['Avg_Distance', 'Avg_Time', 'Bike_User']].corr()
    print(correlation[modal_columns].loc[['Avg_Distance', 'Avg_Time', 'Bike_User']])

# Add this to your main execution block
print_summary_statistics(combined_df)

def plot_overall_modal_split(df):
    modal_columns = ['bus', 'car', 'metro', 'tram', 'cycling', 'train', 'foot']
    overall_modal_split = df[modal_columns].mean()
    
    plt.figure(figsize=(10, 6))
    overall_modal_split.plot(kind='bar')
    plt.title('Overall Modal Split', fontsize=16)
    plt.xlabel('Transport Mode')
    plt.ylabel('Proportion')
    plt.tight_layout()
    plt.savefig("V:\\Prague_Biking\\Data\\Survey Data\\overall_modal_split.png", dpi=300)
    plt.close()

# Add this to your main execution block
plot_overall_modal_split(combined_df)