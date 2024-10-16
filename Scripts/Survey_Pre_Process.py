import pandas as pd
import numpy as np
from datetime import datetime
import json
import geopandas as gpd
from shapely.geometry import Point
from fuzzywuzzy import process
# File path
file_path = r"V:\Prague_Biking\Data\Survey Data\Excel survey results full version (1).xlsx"


neighborhood_dict = {
    'Geo 1': ['Letná', 'Vinohrady', 'Nové Město', 'Josefov', 'Staré Město'],
    'Geo 2': ['Břevnov', 'Petřiny', 'Střešovice', 'Holešovice', 'Bořislavka', 'Dejvice'],
    'Geo 3': ['Podolí', 'Strahov', 'Smíchov', 'Nusle', 'Pankrác'],
    'Geo 4': ['Žižkov', 'Karlín', 'Hloubětín', 'Střížkov', 'Libeň', 'Vysočany', 'Kobylisy'],
    'Geo 5': ['Suchdol']
}

# Flatten the dictionary for easier fuzzy matching
all_neighborhoods = [item for sublist in neighborhood_dict.values() for item in sublist]
# Read the Excel file
df = pd.read_excel(file_path)
def correct_neighborhood(neighborhood):
    if pd.isna(neighborhood):
        return np.nan, np.nan
    
    # Perform fuzzy matching
    match = process.extractOne(neighborhood, all_neighborhoods)
    if match[1] >= 80:  # 80% similarity threshold
        corrected = match[0]
        
        # Find the corresponding geo-area
        for geo, neighborhoods in neighborhood_dict.items():
            if corrected in neighborhoods:
                return corrected, geo
    
    return neighborhood, np.nan  # Return original if no match found
def preprocess_data(df):
    # Debugging: Print column names and data types
    print("Column names:", df.columns.tolist())
    print("\nData types:\n", df.dtypes)
    
    # Step 1: Basic Data Cleaning
    df = df.replace({'': np.nan, ' ': np.nan})  # Replace empty strings and spaces with NaN
    df = df.dropna(how='all')  # Drop rows that are all NaN
    world_shapefile_path = r"V:\Prague_Biking\Data\Maps\ne_110m_admin_0_countries.shp"
    df = process_ip_locations(df, world_shapefile_path)
    # Step 2: Handle dates
    date_columns = ['StartDate', 'EndDate', 'RecordedDate']
    for col in date_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')

    # Step 3: Clean text responses
    text_columns = ['Q7', 'Q9', 'Q11', 'Q14_9_TEXT', 'Q17_9_TEXT', 'Q21_10_TEXT', 'Q22_10_TEXT', 'Q28_7_TEXT', 'Q29_6_TEXT', 'Q33_6_TEXT', 'Q34']
    for col in text_columns:
        if col in df.columns:
            df[col] = df[col].str.strip().str.lower()
    
    if 'Q7' in df.columns:
        df['Q7_corrected'], df['Q7a'] = zip(*df['Q7'].apply(correct_neighborhood))
    
    if 'Q11' in df.columns:
        df['Q11_corrected'], df['Q8a'] = zip(*df['Q11'].apply(correct_neighborhood))
    # Step 4: Identify potential print surveys
    df['potential_print_survey'] = df['RecordedDate'].dt.date == datetime(2024, 10, 8).date()

    # Step 5: Handle conditional questions
    conditional_pairs = [
        ('Q6', 'Q7'),  # If lives in Prague, neighborhood
        ('Q8', 'Q9'),  # If works in Prague, work neighborhood
        ('Q10', 'Q11'),  # If studies in Prague, study neighborhood
    ]
    for condition_q, dependent_q in conditional_pairs:
        if condition_q in df.columns and dependent_q in df.columns:
            mask = (df[condition_q] == 'No') & df[dependent_q].notna()
            df.loc[mask, 'data_inconsistency'] = True

    # Step 6: Encode multiple-choice questions
    multiple_choice_columns = ['Q4', 'Q14', 'Q17', 'Q20', 'Q21', 'Q22', 'Q28', 'Q29', 'Q30', 'Q31']
    for col in multiple_choice_columns:
        if col in df.columns:
            dummies = df[col].str.get_dummies(sep=',')
            dummies.columns = [f"{col}_{c}" for c in dummies.columns]
            df = pd.concat([df, dummies], axis=1)

    # Step 7: Age validation
    if 'Q3' in df.columns:
        df['Q3'] = pd.Categorical(df['Q3'], categories=['Under 18', '18 - 24', '25 - 34', '35 - 44', '45 - 54', '55 - 64', '65  or older'], ordered=True)

    # Step 8: Calculate survey duration
    if 'StartDate' in df.columns and 'EndDate' in df.columns:
        df['calculated_duration'] = (df['EndDate'] - df['StartDate']).dt.total_seconds()

    # Step 9: Flag potential data quality issues
    if 'Progress' in df.columns:
        # Convert 'Progress' to numeric, replacing any non-numeric values with NaN
        df['Progress'] = pd.to_numeric(df['Progress'], errors='coerce')
        
        df['potential_issue'] = np.where(
            (df['Progress'] < 100) |  # Incomplete surveys
            (df.get('calculated_duration', pd.Series()) < 60) |  # Very quick responses
            (df.get('data_inconsistency', pd.Series()) == True),  # Inconsistent conditional responses
            True, False
        )
    else:
        print("Warning: 'Progress' column not found in the dataset.")

    # Step 10: Handle language differences
    if 'UserLanguage' in df.columns:
        df['survey_language'] = np.where(df['UserLanguage'] == 'CS', 'CS', 'EN')
    else:
        print("Warning: 'UserLanguage' column not found in the dataset.")

    # Step 11: Process Likert scale questions
    likert_questions = ['Q24', 'Q29', 'Q26', 'Q27']
    for q in likert_questions:
        if q in df.columns:
            df[q] = pd.Categorical(df[q], categories=['Strongly agree', 'Agree', 'Disagree', 'Strongly disagree'], ordered=True)

    # Step 12: Process slider question
    if 'Q24' in df.columns:
        df['Q24'] = pd.to_numeric(df['Q24'], errors='coerce')

    # Debugging: Print column names and data types after processing
    print("\nColumn names after processing:", df.columns.tolist())
    print("\nData types after processing:\n", df.dtypes)

    return df
def process_ip_locations(df, world_shapefile_path):
    # Check if required columns exist
    required_columns = ['IPAddress', 'LocationLatitude', 'LocationLongitude']
    if not all(col in df.columns for col in required_columns):
        print("Error: One or more required columns not found in the dataset.")
        return df
    
    # Read world shapefile
    world = gpd.read_file(world_shapefile_path)
    
    # Try to identify the country column
    country_column = next((col for col in ['COUNTRY', 'NAME', 'ADMIN', 'SOVEREIGNT'] if col in world.columns), None)
    if country_column is None:
        print("Error: Could not identify a suitable country column in the world shapefile.")
        return df
    
    print(f"Using '{country_column}' as the country column.")
    
    # Convert longitude and latitude to numeric, replacing any non-numeric values with NaN
    df['LocationLongitude'] = pd.to_numeric(df['LocationLongitude'], errors='coerce')
    df['LocationLatitude'] = pd.to_numeric(df['LocationLatitude'], errors='coerce')
    
    # Create geometry column, filtering out rows with NaN coordinates
    valid_coords = df[['LocationLongitude', 'LocationLatitude']].notna().all(axis=1)
    geometry = [Point(xy) for xy in zip(df.loc[valid_coords, 'LocationLongitude'], df.loc[valid_coords, 'LocationLatitude'])]
    gdf = gpd.GeoDataFrame(df[valid_coords], geometry=geometry, crs="EPSG:4326")
    
    # Perform spatial join
    gdf_with_country = gpd.sjoin(gdf, world[['geometry', country_column]], how="left", predicate="within")
    
    # Categorize as Offline (Netherlands) or Online (rest)
    def categorize_survey_type(country, ip):
        if pd.isna(country) or pd.isna(ip) or ip.lower() in ['anon', 'anonymous', 'na', 'n/a', '']:
            return "Unknown"
        elif country == "Netherlands":
            return "Offline"
        else:
            return "Online"
    
    gdf_with_country['survey_type'] = gdf_with_country.apply(lambda row: categorize_survey_type(row[country_column], row['IPAddress']), axis=1)
    
    # Merge the results back to the original dataframe
    df = df.merge(gdf_with_country[['survey_type']], left_index=True, right_index=True, how='left')
    df['survey_type'] = df['survey_type'].fillna('Unknown')
    
    # Categorize based on survey number
    df['survey_version'] = np.where(df.index <= 508, '508', 'Rest')
    
    return df
processed_df = preprocess_data(df)

# Split and save the data into four categories
categories = {
    'Online_508': (processed_df['survey_type'] == 'Online') & (processed_df['survey_version'] == '508'),
    'Offline_508': (processed_df['survey_type'] == 'Offline') & (processed_df['survey_version'] == '508'),
    'Online_Rest': (processed_df['survey_type'] == 'Online') & (processed_df['survey_version'] == 'Rest'),
    'Offline_Rest': (processed_df['survey_type'] == 'Offline') & (processed_df['survey_version'] == 'Rest')
}

for category, mask in categories.items():
    output_path = f"V:\\Prague_Biking\\Data\\Survey Data\\Processed_Cycling_in_Prague_Survey_{category}.xlsx"
    processed_df[mask].to_excel(output_path, index=False)
    print(f"Processed data for {category} saved to {output_path}")

# Save the full processed data
full_output_path = r"V:\Prague_Biking\Data\Survey Data\Processed_Cycling_in_Prague_Survey_Full.xlsx"
processed_df.to_excel(full_output_path, index=False)
print(f"Full processed data saved to {full_output_path}")
