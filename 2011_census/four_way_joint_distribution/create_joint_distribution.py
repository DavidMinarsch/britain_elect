import pandas as pd
import numpy as np
import os

script_dir = os.path.dirname(os.path.abspath(__file__))

# https://www.nomisweb.co.uk/query
###
# Assemble joint distribution of: region - sex - age - ethnicity
###
census_11_male_white = pd.read_csv(script_dir + '/male_white.csv')
census_11_male_asian = pd.read_csv(script_dir + '/male_asian.csv')
census_11_male_black = pd.read_csv(script_dir + '/male_black.csv')
census_11_male_mixed = pd.read_csv(script_dir + '/male_mixed.csv')
census_11_male_other = pd.read_csv(script_dir + '/male_other.csv')
census_11_female_white = pd.read_csv(script_dir + '/female_white.csv')
census_11_female_asian = pd.read_csv(script_dir + '/female_asian.csv')
census_11_female_black = pd.read_csv(script_dir + '/female_black.csv')
census_11_female_mixed = pd.read_csv(script_dir + '/female_mixed.csv')
census_11_female_other = pd.read_csv(script_dir + '/female_other.csv')

# For now drop low ages:
census_11_male_white = census_11_male_white[6:][:].reset_index(drop=True)
census_11_male_asian = census_11_male_asian[6:][:].reset_index(drop=True)
census_11_male_black = census_11_male_black[6:][:].reset_index(drop=True)
census_11_male_mixed = census_11_male_mixed[6:][:].reset_index(drop=True)
census_11_male_other = census_11_male_other[6:][:].reset_index(drop=True)
census_11_female_white = census_11_female_white[6:][:].reset_index(drop=True)
census_11_female_asian = census_11_female_asian[6:][:].reset_index(drop=True)
census_11_female_black = census_11_female_black[6:][:].reset_index(drop=True)
census_11_female_mixed = census_11_female_mixed[6:][:].reset_index(drop=True)
census_11_female_other = census_11_female_other[6:][:].reset_index(drop=True)

n_sex = 2
n_ethnicity = 5
n_region = 10
n_age = 15
census_11 = pd.DataFrame(columns={'sex', 'ethnicity', 'region', 'age', 'N'})
census_11_str = pd.DataFrame(columns={'sex', 'ethnicity', 'region', 'age', 'N'})
for s in range(0, n_sex):
    for e in range(0, n_ethnicity):
        for r in range(0, n_region):
            for a in range(0, n_age):
                if s == 0:
                    sex = 'male'
                else:
                    sex = 'female'
                if e == 0:
                    ethnicity = 'white'
                elif e == 1:
                    ethnicity = 'mixed'
                elif e == 2:
                    ethnicity = 'asian'
                elif e == 3:
                    ethnicity = 'black'
                else:
                    ethnicity = 'other'
                selection = globals()['census_11_' + sex + '_' + ethnicity]
                val = selection.iloc[a, r + 1]
                if not isinstance(val, np.int64):
                    val = int(val.replace(',', ''))
                data = pd.DataFrame({'sex': [s],
                                     'ethnicity': [e],
                                     'region': [r],
                                     'age': [a],
                                     'N': [val]})
                census_11 = census_11.append(data)
                region = selection.columns[r + 1]
                age = selection.iloc[a, 0]
                data = pd.DataFrame({'sex': [sex],
                                     'ethnicity': [ethnicity],
                                     'region': [region],
                                     'age': [age],
                                     'N': [val]})
                census_11_str = census_11_str.append(data)

replace_map = {'East': 'East of England', 'Yorkshire and The Humber': 'Yorkshire and the Humber',
               'East Midlands': 'East Midlands', 'London': 'London', 'North East': 'North East',
               'North West': 'North West', 'South East': 'South East', 'South West': 'South West',
               'Wales': 'Wales', 'West Midlands': 'West Midlands'}
census_11_str['region'] = census_11_str['region'].apply(lambda x: replace_map[x])

census_11 = census_11.astype(int)
census_11_str.N = census_11_str.N.astype(int)
census_11.to_csv(script_dir + '/four_way_joint_distribution.csv', index=False)
census_11_str.to_csv(script_dir + '/four_way_joint_distribution_str.csv', index=False)
