"""
import os
os.chdir('/Users/davidminarsch/Desktop/PythonMLM/Election_Ex_Britain')
exec(open("election15_britain.py").read())
"""
import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
import seaborn as sns; sns.set_context('notebook')
import pystan
from collections import OrderedDict
import pickle
from pystan import StanModel
import plot_coefficients as pc

"""Multilevel Modeling with Poststratification (MRP)"""
# Use multilevel regression to model individual survey responses as a function of demographic and geographic
# predictors, partially pooling respondents across states/regions to an extent determined by the data.
# The final step is poststratification.

"""Step 1: gather national opinion polls (they need to include respondent information down to the level of disaggregation
the analysis is targetting) """

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
#poll_waves_15_subset = poll_waves_15[['id', 'gor', 'education', 'generalElectionVote', 'partyIdW4', 'profile_ethnicity',
#    'gender', 'leftRight', 'profile_past_vote_2010', 'profile_gross_household', 'Age', 'ageGroup', 'enddate', 'wt_full_W5']]
poll_waves_15_subset = poll_waves_15[['id', 'gor', 'generalElectionVote', 'profile_ethnicity', 'gender', 'profile_past_vote_2010', 'profile_gross_household', 'Age', 'ageGroup', 'enddate', 'wt_full_W5']]
poll_waves_15_subset = poll_waves_15_subset.dropna()
# [26652 rows x 14 columns]
poll_waves_15_subset_2 = poll_waves_15_subset[['id', 'gor', 'education', 'generalElectionVote', 'profile_ethnicity',
    'gender', 'profile_past_vote_2010', 'Age', 'enddate']]
poll_waves_15_subset_2["sex"] = poll_waves_15_subset_2["gender"]
ethnicity_map = {"White British" : "White", "White and Black Caribbean": "Mixed_multiple_ethnic_groups", "Any other white background": "White", "Any other black background": "Black_African_Caribbean_Black_British",
    "White and Asian": "Mixed_multiple_ethnic_groups", "Black African": "Black_African_Caribbean_Black_British", "Indian": "Asian_Asian_British", "Chinese": "Asian_Asian_British",
    "Black Caribbean": "Black_African_Caribbean_Black_British", "Other ethnic group": "Other", "Any other mixed background": "Mixed_multiple_ethnic_groups",
    "Pakistani": "Asian_Asian_British", "Any other Asian background": "Asian_Asian_British", "Bangladeshi": "Asian_Asian_British",
    "White and Black African": "Mixed_multiple_ethnic_groups", "Prefer not to say": ""}
poll_waves_15_subset_2["ethnicity"] = poll_waves_15_subset_2["profile_ethnicity"].apply(lambda x: ethnicity_map[x])
poll_waves_15_subset_2 = poll_waves_15_subset_2.drop(["profile_ethnicity"], axis=1)
poll_waves_15_subset_2 = poll_waves_15_subset_2[poll_waves_15_subset_2["ethnicity"] != ""]
sex_map = {"White British" : "White", "White and Black Caribbean": "Mixed_multiple_ethnic_groups", "Any other white background": "White", "Any other black background": "Black_African_Caribbean_Black_British",
    "White and Asian": "Mixed_multiple_ethnic_groups", "Black African": "Black_African_Caribbean_Black_British", "Indian": "Asian_Asian_British", "Chinese": "Asian_Asian_British",
    "Black Caribbean": "Black_African_Caribbean_Black_British", "Other ethnic group": "Other", "Any other mixed background": "Mixed_multiple_ethnic_groups",
    "Pakistani": "Asian_Asian_British", "Any other Asian background": "Asian_Asian_British", "Bangladeshi": "Asian_Asian_British",
    "White and Black African": "Mixed_multiple_ethnic_groups", "Prefer not to say": ""}
age_map = "Age_0_4"  Age_5_7  Age_8_9  Age_10_14  Age_15  Age_16_17
# [26274 rows x 9 columns]

# Drop unnessary columns: 
poll_subset = polls_subset.drop(['org', 'year', 'survey', 'region', 'not_dc', 'state_abbr', 'weight', 'female', 'black'], axis=1)
polls_subset['main'] = np.where(polls_subset['bush'] == 1, 1, np.where(polls_subset['bush'] == 0, 1, 0))

# Drop nan in polls_subset.bush
polls_subset_no_nan = polls_subset[polls_subset.bush.notnull()]
polls_subset_no_nan = polls_subset_no_nan.drop(['main'], axis=1)

# define other data summaries
n = len(polls_subset.bush)              # of survey respondents
n_no_nan = len(polls_subset_no_nan.bush)             # of survey respondents
n_sex = max(polls_subset.sex)           # of sex categories
n_race = max(polls_subset.race)         # of race categories
n_age = max(polls_subset.age)           # of age categories
n_edu = max(polls_subset.edu)           # of education categories
n_state = max(polls_subset.state)       # of states

"""Step 2: create a separate dataset of state-level predictors """
#load in 2010 election data as a predictor
#http://www.electoralcommission.org.uk/our-work/our-research/electoral-data
ge_10 = pd.read_csv("/Users/davidminarsch/Desktop/PythonMLM/Election_Ex_Britain/2010_UK_GE_results/GE2010.csv")
ge_10 = ge_10[['Press Association Reference', 'Constituency Name', 'Region', 'Electorate', 'Votes', 'Con', 'Lab', 'LD', 'Grn', 'UKIP', 'SNP', 'DUP', 'SF', 'PC', 'SDLP']]
ge_10['Other'] = ge_10['Votes'] - ge_10.fillna(0)['Con'] - ge_10.fillna(0)['Lab'] - ge_10.fillna(0)['LD'] - ge_10.fillna(0)['Grn'] - ge_10.fillna(0)['UKIP'] - ge_10.fillna(0)['SNP'] - ge_10.fillna(0)['DUP'] - ge_10.fillna(0)['SF'] - ge_10.fillna(0)['PC'] - ge_10.fillna(0)['SDLP']
ge_10_region = ge_10[['Region', 'Electorate', 'Votes', 'Con', 'Lab', 'LD', 'Grn', 'UKIP', 'SNP', 'DUP', 'SF', 'PC', 'SDLP', 'Other']]
ge_10_region = ge_10_region.groupby('Region').sum()
# Included as a measure of previous vote (a state-level predictor). Include a measure of candidate effects as a state-level predictor.

""" Extra Step: Validation Data"""
# load in 2015 election data as a validation check (http://www.electoralcommission.org.uk/our-work/our-research/electoral-data)
ge_15 = pd.read_csv("/Users/davidminarsch/Desktop/PythonMLM/Election_Ex_Britain/2015_UK_GE_results/RESULTS_FOR_ANALYSIS.csv")
ge_15 = ge_15[['Press Association Reference', 'Constituency Name', 'Region', 'Electorate', 'Votes', 'Con', 'Lab', 'LD', 'Grn', 'UKIP', 'SNP', 'DUP', 'SF', 'PC', 'SDLP']]
ge_15['Other'] = ge_15['Votes'].str.replace(',','').astype(float) - ge_15.fillna(0)['Con'] - ge_15.fillna(0)['Lab'] - ge_15.fillna(0)['LD'] - ge_15.fillna(0)['Grn'] - ge_15.fillna(0)['UKIP'] - ge_15.fillna(0)['SNP'] - ge_15.fillna(0)['DUP'] - ge_15.fillna(0)['SF'] - ge_15.fillna(0)['PC'] - ge_15.fillna(0)['SDLP']
ge_15_region = ge_15[['Region', 'Electorate', 'Votes', 'Con', 'Lab', 'LD', 'Grn', 'UKIP', 'SNP', 'DUP', 'SF', 'PC', 'SDLP', 'Other']]
ge_15_region = ge_15_region.groupby('Region').sum()

"""Step 3: Load 1988 census data to enable poststratification."""
#https://www.nomisweb.co.uk/query/construct/submit.asp?menuopt=201&subcomp=
###
#marginals only by region (England and Wales only):
###
census_11_male_white = pd.read_csv("/Users/davidminarsch/Desktop/PythonMLM/Election_Ex_Britain/2011_census/four_way_joint_distribution/male_white.csv")
census_11_male_asian = pd.read_csv("/Users/davidminarsch/Desktop/PythonMLM/Election_Ex_Britain/2011_census/four_way_joint_distribution/male_asian.csv")
census_11_male_black = pd.read_csv("/Users/davidminarsch/Desktop/PythonMLM/Election_Ex_Britain/2011_census/four_way_joint_distribution/male_black.csv")
census_11_male_mixed = pd.read_csv("/Users/davidminarsch/Desktop/PythonMLM/Election_Ex_Britain/2011_census/four_way_joint_distribution/male_mixed.csv")
census_11_male_other = pd.read_csv("/Users/davidminarsch/Desktop/PythonMLM/Election_Ex_Britain/2011_census/four_way_joint_distribution/male_other.csv")
census_11_female_white = pd.read_csv("/Users/davidminarsch/Desktop/PythonMLM/Election_Ex_Britain/2011_census/four_way_joint_distribution/female_white.csv")
census_11_female_asian = pd.read_csv("/Users/davidminarsch/Desktop/PythonMLM/Election_Ex_Britain/2011_census/four_way_joint_distribution/female_asian.csv")
census_11_female_black = pd.read_csv("/Users/davidminarsch/Desktop/PythonMLM/Election_Ex_Britain/2011_census/four_way_joint_distribution/female_black.csv")
census_11_female_mixed = pd.read_csv("/Users/davidminarsch/Desktop/PythonMLM/Election_Ex_Britain/2011_census/four_way_joint_distribution/female_mixed.csv")
census_11_female_other = pd.read_csv("/Users/davidminarsch/Desktop/PythonMLM/Election_Ex_Britain/2011_census/four_way_joint_distribution/female_other.csv")




###
#raked joint distribution of sex - age - ethnicity - region (England and Wales only):
###
#census_11 = pd.read_csv("/Users/davidminarsch/Desktop/PythonMLM/Election_Ex_Britain/2011_census/joint_distribution_pairs.csv")
#census_11 = census_11.drop(census_11.columns[0], axis=1)
#census_11.rename(columns={'V1': 'id', 'V2': 'N'}, inplace=True)
#census_11["N"] = census_11["N"].apply(pd.to_numeric)
#controversial rounding!:
#census_11["N"] = census_11["N"].apply(lambda x: round(x))
#census_11[["sex", "age", "ethnicity", "region"]] = census_11["id"].apply(lambda x: pd.Series(x.split(".")))
#census_11[["sex", "age", "ethnicity", "region"]] = census_11[["sex", "age", "ethnicity", "region"]].apply(pd.to_numeric)
#census_11 = census_11.drop(["id"], axis=1)

#http://webarchive.nationalarchives.gov.uk/20160105160709/http://www.ons.gov.uk/ons/guide-method/user-guidance/parliamentary-constituencies/data-catalogue-for-parliamentary-constituencies/index.html
prevs_vote
census88 = pd.merge(census88, presvote, on='state', how='left')
# age: categorical variable
# sex: indicator variable
# ethnicity: categorical variable
# region: categorical variable
# N: size of population in this cell

"""Step 4: Fit a regression model for an individual survey response given demographics, geography etc."""
################################
#### 1st model: Probability that a voter casts a vote on a main party candidate
################################
# Pr(Y_i \in {Option_1, Option_2}) = logit^{-1}(alpha[1] + alpha[2] * v_prev_j[i] + a^state_j[i] + a^edu_j[i] + a^sex_j[i] + a^age_j[i]
#    + a^race_j[i] + a^partyID_j[i] + a^ideology_j[i] + a^lastvote_j[i])
# a^{}_j[i] are the varying coefficients associated with each categorical variable; with independent prior distributions:
# a^{}_j[i] ~ N(0,sigma^2_var)
# the variance parameters are assigned a hyper prior distribution:
# sigma^2_var ~ invX^2(v,sigma^2_0)
# with a weak prior specification for v and sigma^2_0

# Model description:
model_1 = """
data {
  int<lower=0> N;
  int<lower=0> n_state;
  int<lower=0> n_edu;
  int<lower=0> n_sex;
  int<lower=0> n_age;
  int<lower=0> n_race;
  #int<lower=0> n_party_id;
  #int<lower=0> n_ideology;
  #int<lower=0> n_lastvote;
  vector[N] state_v_prev;
  int<lower=0,upper=n_state> state[N];
  int<lower=0,upper=n_edu> edu[N];
  int<lower=0,upper=n_sex> sex[N];
  int<lower=0,upper=n_age> age[N];
  int<lower=0,upper=n_race> race[N];
  #int<lower=0,upper=n_party_id> party_id[N];
  #int<lower=0,upper=n_ideology> ideology[N];
  #int<lower=0,upper=n_lastvote> lastvote[N];
  int<lower=0,upper=1> y[N];
} 
parameters {
  vector[2] alpha;
  vector[n_state] a;
  vector[n_edu] b;
  vector[n_sex] c;
  vector[n_age] d;
  vector[n_race] e;
  #vector[n_party_id] f;
  #vector[n_ideology] g;
  #vector[n_lastvote] h;
  real<lower=0,upper=100> sigma_a;
  real<lower=0,upper=100> sigma_b;
  real<lower=0,upper=100> sigma_c;
  real<lower=0,upper=100> sigma_d;
  real<lower=0,upper=100> sigma_e;
  #real<lower=0,upper=100> sigma_f;
  #real<lower=0,upper=100> sigma_g;
  #real<lower=0,upper=100> sigma_h;
  real<lower=0> mu;
  real<lower=0,upper=100> sigma_0;
}
transformed parameters {
  vector[N] y_hat;
  for (i in 1:N)
    y_hat[i] = alpha[1] + alpha[2] * state_v_prev[i] + a[state[i]] + b[edu[i]] + c[sex[i]] + d[age[i]] +
        e[race[i]]; #+ f[party_id[i]] + g[ideology[i]] + h[lastvote[i]];
} 
model {
  a ~ normal (0, sigma_a);
  b ~ normal (0, sigma_b);
  c ~ normal (0, sigma_c);
  d ~ normal (0, sigma_d);
  e ~ normal (0, sigma_e);
  #f ~ normal (0, sigma_f);
  #g ~ normal (0, sigma_g);
  #h ~ normal (0, sigma_h);
  alpha ~ normal(0, 100);
  sigma_a ~ scaled_inv_chi_square(mu,sigma_0);
  sigma_b ~ scaled_inv_chi_square(mu,sigma_0);
  sigma_c ~ scaled_inv_chi_square(mu,sigma_0);
  sigma_d ~ scaled_inv_chi_square(mu,sigma_0);
  sigma_e ~ scaled_inv_chi_square(mu,sigma_0);
  #sigma_f ~ scaled_inv_chi_square(mu,sigma_0);
  #sigma_g ~ scaled_inv_chi_square(mu,sigma_0);
  #sigma_h ~ scaled_inv_chi_square(mu,sigma_0);
  mu ~ uniform(0, 100);
  sigma_0 ~ uniform(0, 100);
  y ~ bernoulli_logit(y_hat);
}
"""

# Model parameters and data:
model_1_data_dict = {'N': n, 'n_state': n_state, 'n_edu': n_edu, 'n_sex': n_sex, 'n_age': n_age, 'n_race': n_race,
  'state': polls_subset.state, 'edu': polls_subset.edu, 'sex': polls_subset.sex, 'age': polls_subset.age,
  'race': polls_subset.race, 'state_v_prev': polls_subset.v_prev, 'y': polls_subset.main}

# Fitting the model:
n_chains = 2
n_iter = 1000
#full_model_fit = pystan.stan(model_code=full_model, data=full_model_data_dict, iter=n_iter, chains=2)
#sm = StanModel(model_code=model_1)
#with open('model_1.pkl', 'wb') as f:
#    pickle.dump(sm, f)
sm = pickle.load(open('model_1.pkl', 'rb'))
model_1_fit = sm.sampling(data=model_1_data_dict, iter=n_iter, chains=n_chains)

# Extract and label parameters:
params_m1 = model_1_fit.extract()
params_m1_alpha_0 = pd.DataFrame({'Intercept' : params_m1['alpha'][:,0]})
params_m1_alpha_1 = pd.DataFrame({'Prev Vote' : params_m1['alpha'][:,1]})
params_m1_a = pd.DataFrame(OrderedDict({'State ' + str(i+1) : params_m1['a'][:,i] for i in range(0,params_m1['a'].shape[1])}))
params_m1_b = pd.DataFrame(OrderedDict({'Edu ' + str(i+1) : params_m1['b'][:,i] for i in range(0,params_m1['b'].shape[1])}))
params_m1_c = pd.DataFrame(OrderedDict({'Sex ' + str(i+1) : params_m1['c'][:,i] for i in range(0,params_m1['c'].shape[1])}))
params_m1_d = pd.DataFrame(OrderedDict({'Age ' + str(i+1) : params_m1['d'][:,i] for i in range(0,params_m1['d'].shape[1])}))
params_m1_e = pd.DataFrame(OrderedDict({'Race ' + str(i+1) : params_m1['e'][:,i] for i in range(0,params_m1['e'].shape[1])}))
params_m1_demo = pd.concat([params_m1_alpha_0, params_m1_b, params_m1_c, params_m1_d, params_m1_e], axis=1)
params_m1_state = pd.concat([params_m1_alpha_1, params_m1_a], axis=1)

# Plot demographic coefficients with confidence intervals:
pc.plot_coefficients(params = params_m1_demo, ticks_list = list(params_m1_demo.columns.values), title = 'Coefficients', f_name = 'DemoCoefficients_ConfidenceIntervals.png')

# Plot state coefficients with confidence intervals:
pc.plot_coefficients(params = params_m1_state, ticks_list = list(params_m1_state.columns.values), title = 'State Intercepts', f_name = 'StateIntercepts_ConfidenceIntervals.png')

# Coefficient Distributions and Traceplots:
model_1_fit.plot()
plt.savefig('ParameterDistributions_model_1.png')

################################
#### 2nd model: Probability that a voter casts a vote for Option_1
################################
# 2nd model:
# Pr(Y_i = Option_1 | Y_i \in {Option_1, Option_2}) = logit^{-1}(beta_0 + beta_1 + b^state_j[i] + b^edu_j[i]
#     + b^sex_j[i] + b^age_j[i] + b^race_j[i] + b^partyID_j[i] + b^ideology_j[i] + b^lastvote_j[i])
# b^{}_j[i] ~ N(0,eta^2_var)
# eta^2_var ~ invX^2(mu,eta^2_0)
# run daily with four-dat moving window(t, t-1, t-2, t-3)

# Model description:
model_2 = """
data {
  int<lower=0> N;
  int<lower=0> n_state;
  int<lower=0> n_edu;
  int<lower=0> n_sex;
  int<lower=0> n_age;
  int<lower=0> n_race;
  #int<lower=0> n_party_id;
  #int<lower=0> n_ideology;
  #int<lower=0> n_lastvote;
  vector[N] state_v_prev;
  int<lower=0,upper=n_state> state[N];
  int<lower=0,upper=n_edu> edu[N];
  int<lower=0,upper=n_sex> sex[N];
  int<lower=0,upper=n_age> age[N];
  int<lower=0,upper=n_race> race[N];
  #int<lower=0,upper=n_party_id> party_id[N];
  #int<lower=0,upper=n_ideology> ideology[N];
  #int<lower=0,upper=n_lastvote> lastvote[N];
  int<lower=0,upper=1> y[N];
} 
parameters {
  vector[2] alpha;
  vector[n_state] a;
  vector[n_edu] b;
  vector[n_sex] c;
  vector[n_age] d;
  vector[n_race] e;
  #vector[n_party_id] f;
  #vector[n_ideology] g;
  #vector[n_lastvote] h;
  real<lower=0,upper=100> sigma_a;
  real<lower=0,upper=100> sigma_b;
  real<lower=0,upper=100> sigma_c;
  real<lower=0,upper=100> sigma_d;
  real<lower=0,upper=100> sigma_e;
  #real<lower=0,upper=100> sigma_f;
  #real<lower=0,upper=100> sigma_g;
  #real<lower=0,upper=100> sigma_h;
  real<lower=0> mu;
  real<lower=0,upper=100> sigma_0;
}
transformed parameters {
  vector[N] y_hat;
  for (i in 1:N)
    y_hat[i] = alpha[1] + alpha[2] * state_v_prev[i] + a[state[i]] + b[edu[i]] + c[sex[i]] + d[age[i]] + e[race[i]];
    #+ f[party_id[i]] + g[ideology[i]] + h[lastvote[i]];
} 
model {
  a ~ normal (0, sigma_a);
  b ~ normal (0, sigma_b);
  c ~ normal (0, sigma_c);
  d ~ normal (0, sigma_d);
  e ~ normal (0, sigma_e);
  #f ~ normal (0, sigma_f);
  #g ~ normal (0, sigma_g);
  #h ~ normal (0, sigma_h);
  alpha ~ normal(0, 100);
  sigma_a ~ scaled_inv_chi_square(mu,sigma_0);
  sigma_b ~ scaled_inv_chi_square(mu,sigma_0);
  sigma_c ~ scaled_inv_chi_square(mu,sigma_0);
  sigma_d ~ scaled_inv_chi_square(mu,sigma_0);
  sigma_e ~ scaled_inv_chi_square(mu,sigma_0);
  #sigma_f ~ scaled_inv_chi_square(mu,sigma_0);
  #sigma_g ~ scaled_inv_chi_square(mu,sigma_0);
  #sigma_h ~ scaled_inv_chi_square(mu,sigma_0);
  mu ~ uniform(0, 100);
  sigma_0 ~ uniform(0, 100);
  y ~ bernoulli_logit(y_hat);
}
"""

# Model parameters and data:
model_2_data_dict = {'N': n_no_nan, 'n_state': n_state, 'n_edu': n_edu, 'n_sex': n_sex, 'n_age': n_age, 'n_race': n_race,
  'state': polls_subset_no_nan.state, 'edu': polls_subset_no_nan.edu, 'sex': polls_subset_no_nan.sex, 'age': polls_subset_no_nan.age,
  'race': polls_subset_no_nan.race, 'state_v_prev': polls_subset_no_nan.v_prev, 'y': polls_subset_no_nan.bush.astype(int)}

# Fitting the model:
n_chains = 2
n_iter = 1000
#full_model_fit = pystan.stan(model_code=full_model, data=full_model_data_dict, iter=n_iter, chains=2)
#sm = StanModel(model_code=model_2)
#with open('model_2.pkl', 'wb') as f:
#    pickle.dump(sm, f)
sm = pickle.load(open('model_2.pkl', 'rb'))
model_2_fit = sm.sampling(data=model_2_data_dict, iter=n_iter, chains=n_chains)


# Extract and label parameters:
params_m2 = model_2_fit.extract()
params_m2_alpha_0 = pd.DataFrame({'Intercept' : params_m2['alpha'][:,0]})
params_m2_alpha_1 = pd.DataFrame({'Prev Vote' : params_m2['alpha'][:,1]})
params_m2_a = pd.DataFrame(OrderedDict({'State ' + str(i+1) : params_m2['a'][:,i] for i in range(0,params_m2['a'].shape[1])}))
params_m2_b = pd.DataFrame(OrderedDict({'Edu ' + str(i+1) : params_m2['b'][:,i] for i in range(0,params_m2['b'].shape[1])}))
params_m2_c = pd.DataFrame(OrderedDict({'Sex ' + str(i+1) : params_m2['c'][:,i] for i in range(0,params_m2['c'].shape[1])}))
params_m2_d = pd.DataFrame(OrderedDict({'Age ' + str(i+1) : params_m2['d'][:,i] for i in range(0,params_m2['d'].shape[1])}))
params_m2_e = pd.DataFrame(OrderedDict({'Race ' + str(i+1) : params_m2['e'][:,i] for i in range(0,params_m2['e'].shape[1])}))
params_m2_demo = pd.concat([params_m2_alpha_0, params_m2_b, params_m2_c, params_m2_d, params_m2_e], axis=1)
params_m2_state = pd.concat([params_m2_alpha_1, params_m2_a], axis=1)

# Plot coefficients with confidence intervals:
pc.plot_coefficients(params = params_m2_demo, ticks_list = list(params_m2_demo.columns.values), title = 'Coefficients', f_name = 'DemoCoefficients_ConfidenceIntervals_m2.png')

# Plot coefficients with confidence intervals:
pc.plot_coefficients(params = params_m2_state, ticks_list = list(params_m2_state.columns.values), title = 'State Intercepts', f_name = 'StateIntercepts_ConfidenceIntervals_m2.png')

# Traceplot:
model_2_fit.plot()
plt.savefig('ParameterDistributions_model_2.png')
plt.show()

# Plot individual parameter's different chains:
"""b = basic_model_fit.extract(permuted=True)['b']
b_split = np.array_split(b, n_chains) # assumes that the b array is just one chain tacked onto the end of another
for i in range(n_chains):
    plt.plot(b_split[i])
plt.savefig('Traceplot.png')
plt.show()"""

"""5. Poststratification"""
## Using the model inferences to estimate avg opinion for each state
# construct the n.sims x 3264 matrix
alpha_m1 = pd.DataFrame(params_m1['alpha'])
a_m1 = pd.DataFrame(params_m1['a'])
b_m1 = pd.DataFrame(params_m1['b'])
c_m1 = pd.DataFrame(params_m1['c'])
d_m1 = pd.DataFrame(params_m1['d'])
e_m1 = pd.DataFrame(params_m1['e'])
alpha_m2 = pd.DataFrame(params_m2['alpha'])
a_m2 = pd.DataFrame(params_m2['a'])
b_m2 = pd.DataFrame(params_m2['b'])
c_m2 = pd.DataFrame(params_m2['c'])
d_m2 = pd.DataFrame(params_m2['d'])
e_m2 = pd.DataFrame(params_m2['e'])
L = census88.shape[0]
y_pred = np.full((int((n_iter / 2) * n_chains),L), np.nan)
y_pred_cond = np.full((int((n_iter / 2) * n_chains),L), np.nan)
for l in range(0, L):
  y_pred[:,l] = sp.special.expit(alpha_m1.ix[:,0] + alpha_m1.ix[:,1] * census88.v_prev[l] + 
    a_m1.ix[:,census88.state[l]-1] + b_m1.ix[:,census88.edu[l]-1] + c_m1.ix[:,census88.sex[l]-1] + 
    d_m1.ix[:,census88.age[l]-1] + e_m1.ix[:,census88.race[l]-1])
  y_pred_cond[:,l] = sp.special.expit(alpha_m2.ix[:,0] + alpha_m2.ix[:,1] * census88.v_prev[l] + 
    a_m2.ix[:,census88.state[l]-1] + b_m2.ix[:,census88.edu[l]-1] + c_m2.ix[:,census88.sex[l]-1] + 
    d_m2.ix[:,census88.age[l]-1] + e_m2.ix[:,census88.race[l]-1])

# Convert to unconditional probabilities:
y_bush = y_pred_cond * y_pred
y_non_bush = (1 - y_pred_cond) * y_pred
y_non = (1 - y_pred)

# Normalized:
y_bush_norm = y_bush / (y_bush + y_non_bush)
y_non_bush_norm = y_non_bush / (y_bush + y_non_bush)

# average over strata within each state
y_pred_state = np.full((int((n_iter / 2) * n_chains),n_state), np.nan)
for j in range(1,n_state+1):
    sel = [s for s in range(L) if census88.state[s] ==  j]
    y_pred_state[:,j-1] = np.divide((np.dot(y_bush_norm[:,sel],(census88[census88.state == j]).N)),sum((census88[census88.state == j]).N))
y_pred_state = pd.DataFrame(y_pred_state)

y_pred_state_bush = np.full((int((n_iter / 2) * n_chains),n_state), np.nan)
for j in range(1,n_state+1):
    sel = [s for s in range(L) if census88.state[s] ==  j]
    y_pred_state_bush[:,j-1] = np.divide((np.dot(y_bush[:,sel],(census88[census88.state == j]).N)),sum((census88[census88.state == j]).N))
y_pred_state_bush = pd.DataFrame(y_pred_state_bush)

y_pred_state_non_bush = np.full((int((n_iter / 2) * n_chains),n_state), np.nan)
for j in range(1,n_state+1):
    sel = [s for s in range(L) if census88.state[s] ==  j]
    y_pred_state_non_bush[:,j-1] = np.divide((np.dot(y_non_bush[:,sel],(census88[census88.state == j]).N)),sum((census88[census88.state == j]).N))
y_pred_state_non_bush = pd.DataFrame(y_pred_state_non_bush)

y_pred_state_non = np.full((int((n_iter / 2) * n_chains),n_state), np.nan)
for j in range(1,n_state+1):
    sel = [s for s in range(L) if census88.state[s] ==  j]
    y_pred_state_non[:,j-1] = np.divide((np.dot(y_non[:,sel],(census88[census88.state == j]).N)),sum((census88[census88.state == j]).N))
y_pred_state_non = pd.DataFrame(y_pred_state_non)

"""#Old plotting method:
plt.figure(figsize=(16, 6))
sns.boxplot(data=y_pred_state, whis=np.inf, color="c")
plt.savefig('Estimates_state.png')
plt.show()"""

# New plotting method:
ticks_list = list(state_info.state_abbr.values)
plt.figure(figsize=(10,20))
plt.plot(y_pred_state.median(), range(y_pred_state.shape[1]), 'ko', ms = 10)
plt.plot(election88.electionresult, range(election88.shape[0]), 'r.', ms = 10)
plt.hlines(range(y_pred_state.shape[1]), y_pred_state.quantile(0.025), y_pred_state.quantile(0.975), 'k')
plt.hlines(range(y_pred_state.shape[1]), y_pred_state.quantile(0.25), y_pred_state.quantile(0.75), 'k', linewidth = 3)
plt.axvline(0.5, linestyle = 'dashed', color = 'k')
plt.xlabel('Median State Estimate (50 and 95% CI) and Actual Election Outcome (red)')
plt.yticks(range(y_pred_state.shape[1]), ticks_list)
plt.ylim([-1, y_pred_state.shape[1]])
plt.xlim([(min(y_pred_state.quantile(0.025))-0.5), (max(y_pred_state.quantile(0.975))+0.5)])
plt.title('State Estimates')
plt.tight_layout()
plt.savefig('State_Estimates_Normalized.png')

# New plotting method:
ticks_list = list(state_info.state_abbr.values)
plt.figure(figsize=(10,20))
plt.plot(y_pred_state_bush.median(), range(y_pred_state_bush.shape[1]), 'ro', ms = 10)
plt.plot(y_pred_state_non_bush.median(), range(y_pred_state_non_bush.shape[1]), 'bo', ms = 10)
plt.plot(y_pred_state_non.median(), range(y_pred_state_non.shape[1]), 'yo', ms = 10)
plt.plot(election88.electionresult, range(election88.shape[0]), 'm.', ms = 10)
plt.hlines(range(y_pred_state_bush.shape[1]), y_pred_state_bush.quantile(0.025), y_pred_state_bush.quantile(0.975), 'r')
plt.hlines(range(y_pred_state_bush.shape[1]), y_pred_state_bush.quantile(0.25), y_pred_state_bush.quantile(0.75), 'r', linewidth = 3)
plt.hlines(range(y_pred_state_non.shape[1]), y_pred_state_non.quantile(0.025), y_pred_state_non.quantile(0.975), 'y')
plt.hlines(range(y_pred_state_non.shape[1]), y_pred_state_non.quantile(0.25), y_pred_state_non.quantile(0.75), 'y', linewidth = 3)
plt.hlines(range(y_pred_state_non_bush.shape[1]), y_pred_state_non_bush.quantile(0.025), y_pred_state_non_bush.quantile(0.975), 'b')
plt.hlines(range(y_pred_state_non_bush.shape[1]), y_pred_state_non_bush.quantile(0.25), y_pred_state_non_bush.quantile(0.75), 'b', linewidth = 3)
plt.axvline(0.5, linestyle = 'dashed', color = 'k')
plt.xlabel('Median State Estimate (50 and 95% CI) and Actual Election Outcome (red)')
plt.yticks(range(y_pred_state_bush.shape[1]), ticks_list)
plt.ylim([-1, y_pred_state_bush.shape[1]])
plt.xlim([0,1])
#plt.xlim([(min(y_pred_state_bush.quantile(0.025))-0.5), (max(y_pred_state_bush.quantile(0.975))+0.5)])
plt.title('State Estimates')
plt.tight_layout()
plt.savefig('State_Estimates_Actual.png')

#"""Extension: A more intricate model"""
#extended_model_fit = pystan.stan(file='election88_expansion.stan', data=full_model_data_dict, iter=1000, chains=4)

"""Census Data US:
http://dataferrett.census.gov/

National Election Study

ideology and ethnicity are key!"""