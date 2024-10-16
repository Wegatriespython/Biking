import pandas as pd
from Data_Analysis_Tools import load_data, create_generic_columns, print_nan_info, analyze_distributions, analyze_cross_tabulations, analyze_likert_scales


def main(tee: bool = True):
    file_path = r"V:\Prague_Biking\Data\Survey Data\Excel survey results full version (1).xlsx"
    df = load_data(file_path, skiprows=[1])
    #Skip the first row because it's a duplicate of the column names

    column_mappings = {
        'commute_modality': ('Q17', 'Q14'),
        'commute_distance': ('Q15', 'Q12'),
        'commute_time': ('Q16', 'Q13')
    }
    df = create_generic_columns(df, column_mappings)
    df['Bike_User'] = ((df['Q18'] == 'Yes') | (df['Q19'] == 'Yes')).astype(int)

    columns_to_check = [
        'Q3', 'Q4', 'Q5', 'Q6', 'Q7a', 'Q8', 'Q10', 'Q15', 'Q17', 'Q18', 'Q19', 'Q20', 'Q21', 'Q22',
        'Q23_1', 'Q28', 'Q29', 'Q30', 'Q31', 'Q32', 'Q33', 'commute_modality', 'commute_distance', 'commute_time'
    ]

    if tee:
        print(f"Shape of dataframe: {df.shape}")
    nan_info = print_nan_info(df, columns_to_check, tee)

    distributions = [
        ('Q3', "Age Distribution"),
        ('Q4', "Gender Distribution"),
        ('Q7a', "Geographic Distribution across Prague"),
        ('Q5', "Employment Status Distribution"),
        ('commute_modality', "Most Used Commuting Types", True),
        ('Q18', "Bicycle Ownership"),
        ('Q19', "Bike-Sharing Service Usage"),
        ('Q6', "Living in Prague"),
        ('Q8', "Working in Prague"),
        ('Q10', "Studying in Prague"),
        ('Q21', "Encouragement for Cycling", True),
        ('Q22', "Discouragement for Cycling", True),
        ('Q28', "Benefits of Cycling", True),
        ('Q30', "Prague Infrastructure Suitability"),
        ('Q31', "Prague City Hall 1"),
        ('Q32', "Prague City Hall 2"),
        ('Q33', "Policy Choices", True)
    ]
    distribution_results = analyze_distributions(df, distributions, tee)

    cross_tabs = [
        ('Q5', 'Q3', "Employment Status vs Age Group"),
        ('Q20', 'Q4', "Cycling Frequency vs Gender"),
        ('Q3', 'commute_modality', "Age vs Commuting Types", True),
        ('Q4', 'commute_modality', "Gender vs Commuting Types", True),
        ('Q3', 'Q20', "Age vs Cycling Frequency"),
        ('Q4', 'Q20', "Gender vs Cycling Frequency"),
        ('commute_distance', 'commute_modality', "Commute Distance vs Commuting Types", True),
        ('commute_distance', 'Q20', "Commute Distance vs Cycling Frequency"),
        ('Q5', 'commute_modality', "Employment Status vs Commuting Types", True),
        ('Q5', 'Q20', "Employment Status vs Cycling Frequency"),
        ('Bike_User', 'Q21', "Bike Users vs Encouragement", True),
        ('Bike_User', 'Q22', "Bike Users vs Discouragement", True),
        ('Q5', 'Q6', "Employment Status vs Living in Prague"),
        ('Q5', 'Q8', "Employment Status vs Working in Prague"),
        ('Q5', 'Q10', "Employment Status vs Studying in Prague"),
        ('Q5', 'Q20', "Employment Status vs Cycling Frequency")
    ]
    cross_tab_results = analyze_cross_tabulations(df, cross_tabs, tee)

    likert_scales = [
        ('Q23_1', "Likert Scale: Barrier Removal"),
        ('Q30', "Likert Scale: Prague Infrastructure Suitability"),
        ('Q31', "Likert Scale: Prague City Hall 1"),
        ('Q32', "Likert Scale: Prague City Hall 2")
    ]
    likert_results = analyze_likert_scales(df, likert_scales, tee)

    if tee:
        print("\nVerification complete. Data distributions and cross-tabulations have been displayed as tables.")

    return {
        'dataframe': df,
        'nan_info': nan_info,
        'distributions': distribution_results,
        'cross_tabs': cross_tab_results,
        'likert_scales': likert_results
    }

if __name__ == "__main__":
    main(tee=True)
