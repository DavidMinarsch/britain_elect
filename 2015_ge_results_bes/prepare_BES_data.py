import pandas as pd
# data source: http://www.britishelectionstudy.com/data-object/wave-5-of-the-2014-2017-british-election-study-internet-panel-daily-file/
# March 2015 - May 2015; N = 30,725; Mode: Online survey.
# Selected vars:
# - date of interview: enddate
# - Age (What is your age?)
# [- profile_gross_personal (What is your gross personal income?)]
# - profile_gross_household (What is your gross household income?)
# [- profile_past_vote_2005 (Thinking back to the General Election in May 2005, do you remember which party you voted for then - or perhaps you didn't vote?)]
# - profile_past_vote_2010 (Thinking back to the General Election in May 2010, do you remember which party you voted for then - or perhaps you didn't vote?)
# - leftRight (In politics people sometimes talk of left and right. Where would you place yourself on the following scale?)
# - gender (Are you male or female?)
# - profile_ethnicity (To which of these groups do you consider you belong?)
# - partyId (Generally speaking, do you think of yourself as Labour, Conservative, Liberal Democrat or what?)
# - generalElectionVote (And if there were a UK General Election tomorrow, which party would you vote for?)
# [- generalElectionVoteSqueeze (Which party do you think you are most likely to vote for?)]
# - education (What is the highest educational or work-related qualification you have?)
# - gor (Which area of the UK do you live in?)
"""
poll_waves.gor.isnull().sum()
1
poll_waves.education.isnull().sum()
127
poll_waves.generalElectionVote.isnull().sum()
453
poll_waves.partyIdW4.isnull().sum()
2640
poll_waves.profile_ethnicity.isnull().sum()
20
poll_waves.gender.isnull().sum()
0
poll_waves.leftRight.isnull().sum()
0
poll_waves.profile_past_vote_2010.isnull().sum()
1189
poll_waves.profile_gross_household.isnull().sum()
101
poll_waves.Age.isnull().sum()
0
poll_waves.ageGroup.isnull().sum()
0
poll_waves.enddate.isnull().sum()
0
poll_waves.wt_full_W5.isnull().sum()
0
"""
poll_waves_15 = pd.read_stata('/Users/davidminarsch/Desktop/PythonMLM/Election_Ex_Britain/BES2015_W5_v1.0.dta')
# [30725 rows x 355 columns]
# poll_waves_15_subset = poll_waves_15[['id', 'gor', 'education', 'generalElectionVote', 'partyIdW4', 'profile_ethnicity',
#    'gender', 'leftRight', 'profile_past_vote_2010', 'profile_gross_household', 'Age', 'ageGroup', 'enddate', 'wt_full_W5']]
poll_waves_15_subset = poll_waves_15[['id', 'gor', 'generalElectionVote', 'profile_ethnicity', 'gender', 'Age', 'enddate']]
poll_waves_15 = poll_waves_15.dropna()
poll_waves_15_subset.rename(columns={'gender': 'sex'}, inplace=True)
replace_map = {"Male": "male", "Female": "female"}
poll_waves_15_subset["sex"] = poll_waves_15_subset["sex"].apply(lambda x: replace_map[x])
ethnicity_map = {"White British" : "White", "White and Black Caribbean": "Mixed_multiple_ethnic_groups", "Any other white background": "White", "Any other black background": "Black_African_Caribbean_Black_British",
    "White and Asian": "Mixed_multiple_ethnic_groups", "Black African": "Black_African_Caribbean_Black_British", "Indian": "Asian_Asian_British", "Chinese": "Asian_Asian_British",
    "Black Caribbean": "Black_African_Caribbean_Black_British", "Other ethnic group": "Other", "Any other mixed background": "Mixed_multiple_ethnic_groups",
    "Pakistani": "Asian_Asian_British", "Any other Asian background": "Asian_Asian_British", "Bangladeshi": "Asian_Asian_British",
    "White and Black African": "Mixed_multiple_ethnic_groups", "Prefer not to say": ""}
poll_waves_15_subset["ethnicity"] = poll_waves_15_subset["profile_ethnicity"].apply(lambda x: ethnicity_map[x])
poll_waves_15_subset = poll_waves_15_subset[poll_waves_15_subset["ethnicity"] != ""]
ethnicity_map = {"White" : "white", "Mixed_multiple_ethnic_groups": "mixed", "Black_African_Caribbean_Black_British": "black",
                 "Asian_Asian_British": "asian", "Other": "other"}
poll_waves_15_subset["ethnicity"] = poll_waves_15_subset["ethnicity"].apply(lambda x: ethnicity_map[x])
poll_waves_15_subset = poll_waves_15_subset.drop(["profile_ethnicity"], axis=1)
poll_waves_15_subset = poll_waves_15_subset[poll_waves_15_subset["ethnicity"] != ""]
# Drop Scotland and Northern Ireland (not supported atm):
poll_waves_15_subset = poll_waves_15_subset[poll_waves_15_subset["gor"] != "Scotland"]
poll_waves_15_subset = poll_waves_15_subset[poll_waves_15_subset["gor"] != "Northern Ireland"]
poll_waves_15_subset.rename(columns={'gor': 'region'}, inplace=True)
# Drop non-adults (not supported atm):
age_map = {18: "Age 18 to 19", 19: "Age 18 to 19", 20: "Age 20 to 24", 21: "Age 20 to 24", 22: "Age 20 to 24", 23: "Age 20 to 24",
           24: "Age 20 to 24", 25: "Age 25 to 29", 26: "Age 25 to 29", 27: "Age 25 to 29", 28: "Age 25 to 29", 29: "Age 25 to 29",
           30: "Age 30 to 34", 31: "Age 30 to 34", 32: "Age 30 to 34", 33: "Age 30 to 34", 34: "Age 30 to 34", 35: "Age 35 to 39",
           36: "Age 35 to 39", 37: "Age 35 to 39", 38: "Age 35 to 39", 39: "Age 35 to 39", 40: "Age 40 to 44", 41: "Age 40 to 44",
           42: "Age 40 to 44", 43: "Age 40 to 44", 44: "Age 40 to 44", 45: "Age 45 to 49", 46: "Age 45 to 49", 47: "Age 45 to 49",
           48: "Age 45 to 49", 49: "Age 45 to 49", 50: "Age 50 to 54", 51: "Age 50 to 54", 52: "Age 50 to 54", 53: "Age 50 to 54",
           54: "Age 50 to 54", 55: "Age 55 to 59", 56: "Age 55 to 59", 57: "Age 55 to 59", 58: "Age 55 to 59", 59: "Age 55 to 59",
           60: "Age 60 to 64", 61: "Age 60 to 64", 62: "Age 60 to 64", 63: "Age 60 to 64", 64: "Age 60 to 64", 65: "Age 65 to 69",
           66: "Age 65 to 69", 67: "Age 65 to 69", 68: "Age 65 to 69", 69: "Age 65 to 69", 70: "Age 70 to 74", 71: "Age 70 to 74",
           72: "Age 70 to 74", 73: "Age 70 to 74", 74: "Age 70 to 74", 75: "Age 75 to 79", 76: "Age 75 to 79", 77: "Age 75 to 79",
           78: "Age 75 to 79", 79: "Age 75 to 79", 80: "Age 80 to 84", 81: "Age 80 to 84", 82: "Age 80 to 84", 83: "Age 80 to 84",
           84: "Age 80 to 84", 17: "Age 16 to 17", 16: "Age 16 to 17"}
for i in range(85,121):
    age_map[i] = "Age 85 and over"
for key in list(age_map.keys()):
    if key not in poll_waves_15_subset["Age"].unique():
        age_map.pop(key, None)
poll_waves_15_subset["age"] = poll_waves_15_subset["Age"].apply(lambda x: age_map[x])
poll_waves_15_subset = poll_waves_15_subset[poll_waves_15_subset["age"] != "Age 16 to 17"]
party_map = {"Liberal Democrat": "LD", "Labour": "Lab", "United Kingdom Independence Party (UKIP)": "UKIP", "Conservative" : "Con",
             "Don't know": "Don't know", "Green Party": "Grn", "Other": "Other", "British National Party (BNP)": "Other",
             "Scottish National Party (SNP)": "SNP", "I would not vote": "Don't vote", "Plaid Cymru": "PC"}
poll_waves_15_subset["vote"] = poll_waves_15_subset["generalElectionVote"].apply(lambda x: party_map[x])
poll_waves_15_subset = poll_waves_15_subset.drop(["Age", "id", "generalElectionVote"], axis=1)
poll_waves_15_subset = poll_waves_15_subset.reset_index(drop=True)
# (24579, 7)
poll_waves_15_subset.to_csv("/Users/davidminarsch/Desktop/PythonMLM/Election_Ex_Britain/bes_poll_data.csv", index=False)
