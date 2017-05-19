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
import copy

"""Multilevel Modeling with Poststratification (MRP)"""
# Use multilevel regression to model individual survey responses as a function of demographic and geographic
# predictors, partially pooling respondents across states/regions to an extent determined by the data.
# The final step is poststratification.

sex_categories = {"male": 1, "female": 2}
ethnicity_categories = {"white": 1, "mixed": 2, "black": 3, "asian": 4, "other": 5}
age_categories = {"Age 18 to 19": 1, "Age 20 to 24": 2, "Age 25 to 29": 3, "Age 30 to 34": 4, "Age 35 to 39": 5,
                  "Age 40 to 44": 6, "Age 45 to 49": 7, "Age 50 to 54": 8, "Age 55 to 59": 9, "Age 60 to 64": 10,
                  "Age 65 to 69": 11, "Age 65 to 69": 12, "Age 70 to 74": 13, "Age 75 to 79": 14, "Age 80 to 84": 15,
                  "Age 85 and over": 16}
region_categories = {'East Midlands': 1, 'East of England': 2, 'London': 3, 'North East': 4, 'North West': 5,
                     'South East': 6, 'South West': 7, 'Wales': 8, 'West Midlands': 9, 'Yorkshire and the Humber': 10}
enddate_categories = {'2015-03-30': 1, '2015-03-31': 2, '2015-04-01': 3, '2015-04-02': 4, '2015-04-03': 5, '2015-04-04': 6,
                   '2015-04-05': 7, '2015-04-06': 8, '2015-04-07': 9, '2015-04-08': 10, '2015-04-09': 11, '2015-04-10': 12,
                   '2015-04-11': 13, '2015-04-12': 14, '2015-04-13': 15, '2015-04-14': 16, '2015-04-15': 17, '2015-04-16': 18,
                   '2015-04-17': 19, '2015-04-18': 20, '2015-04-19': 21, '2015-04-20': 22, '2015-04-21': 23, '2015-04-22': 24,
                   '2015-04-23': 25, '2015-04-24': 26, '2015-04-25': 27, '2015-04-26': 28, '2015-04-27': 29, '2015-04-28': 30,
                   '2015-04-29': 31, '2015-04-30': 32, '2015-05-01': 33, '2015-05-02': 34, '2015-05-03': 35, '2015-05-04': 36,
                   '2015-05-05': 37, '2015-05-06': 38}
party_categories = {'Con': 1, 'Grn': 2, 'LD': 3, 'Lab': 4, 'Other': 5, 'PC': 6, 'UKIP': 7, "Don't vote": 8, "Don't know": 9}

"""Step 1: gather national opinion polls (they need to include respondent information down to the level of disaggregation
the analysis is targetting) """
polls = pd.read_csv("/Users/davidminarsch/Desktop/PythonMLM/Election_Ex_Britain/bes_poll_data.csv")
# drop SNP voters in regions outside Scottland:
polls = polls[polls['vote'] !='SNP']
# what to do about 'Don't know' voters?
polls['main'] = np.where(polls['vote'] == 'Con', 1, np.where(polls['vote'] == 'Lab', 1, 0))
# polls.shape
# (24579, 6)
# age; n_age = 13
# sex; n_sex = 2
# ethnicity; n_ethnicity = 5
# region; n_region = 10
# enddate
# vote; n_vote = 10
n_age = len(polls.age.unique().tolist())
n_sex = len(polls.sex.unique().tolist())
n_ethnicity = len(polls.ethnicity.unique().tolist())
n_region = len(polls.region.unique().tolist())
n = polls.shape[0]

polls_numeric = copy.deepcopy(polls)
polls_numeric["sex"] = polls_numeric["sex"].apply(lambda x: sex_categories[x])
polls_numeric["age"] = polls_numeric["age"].apply(lambda x: age_categories[x])
polls_numeric["ethnicity"] = polls_numeric["ethnicity"].apply(lambda x: ethnicity_categories[x])
polls_numeric["region"] = polls_numeric["region"].apply(lambda x: region_categories[x])
polls_numeric["enddate"] = polls_numeric["enddate"].apply(lambda x: enddate_categories[x])
polls_numeric["vote"] = polls_numeric["vote"].apply(lambda x: party_categories[x])

"""Step 2: create a separate dataset of region-level predictors """
# load in 2010 election data as a region level predictor
# (http://www.electoralcommission.org.uk/our-work/our-research/electoral-data)
ge_10 = pd.read_csv("/Users/davidminarsch/Desktop/PythonMLM/Election_Ex_Britain/2010_UK_GE_results/GE2010.csv")
ge_10 = ge_10[['Press Association Reference', 'Constituency Name', 'Region', 'Electorate', 'Votes', 'Con', 'Lab', 'LD', 'Grn', 'UKIP', 'SNP','PC']]
ge_10['Other'] = ge_10['Votes'] - ge_10.fillna(0)['Con'] - ge_10.fillna(0)['Lab'] - ge_10.fillna(0)['LD'] - ge_10.fillna(0)['Grn'] - ge_10.fillna(0)['UKIP'] - ge_10.fillna(0)['SNP'] - ge_10.fillna(0)['PC']
ge_10["Don't vote"] = ge_10['Electorate'] - ge_10['Votes']
ge_10_region = ge_10[['Region', 'Electorate', 'Votes', 'Con', 'Lab', 'LD', 'Grn', 'UKIP', 'SNP', 'PC', 'Other', "Don't vote"]]
ge_10_region = ge_10_region.groupby('Region').sum()
ge_10_region = ge_10_region.drop(["SNP"], 1)
ge_10_region = ge_10_region.drop(["Northern Ireland", "Scotland"], 0)
ge_10_region = ge_10_region.rename(index={'Eastern': 'East of England'})
ge_10_region_share = ge_10_region.div(ge_10_region['Votes'], axis=0)
ge_10_region_share['main'] = ge_10_region_share['Con'] + ge_10_region_share['Lab']
# UK parties:
# Conservatives (Con)
# Labour (Lab)
# Liberal Democrats (LD)
# Greens (Grn)
# UK Independence Party (UKIP)
# Scotish National Party (SNP) - Scottland only
# Democratic Unionists (DUP) - Northern Ireland only
# Sinn Fein (SF) - Nothern Ireland only
# Plaid Cymru (PC) - Wales only
# Social Democratic & Labour Party (SDLP) - Northern Ireland only
n_state = ge_10_region.shape[0]

ge_10_region_share = ge_10_region_share.rename(index=region_categories)
ge_10_region_share = ge_10_region_share.rename(columns=party_categories)

""" Extra Step: Validation Data"""
# load in 2015 election data as a validation check
# (http://www.electoralcommission.org.uk/our-work/our-research/electoral-data)
ge_15 = pd.read_csv("/Users/davidminarsch/Desktop/PythonMLM/Election_Ex_Britain/2015_UK_GE_results/RESULTS_FOR_ANALYSIS.csv")
ge_15 = ge_15[['Press Association Reference', 'Constituency Name', 'Region', 'Electorate', 'Votes', 'Con', 'Lab', 'LD', 'Grn', 'UKIP', 'SNP', 'PC']]
ge_15['Votes'] = ge_15['Votes'].str.replace(',','').astype(float) 
ge_15['Electorate'] = ge_15['Electorate'].str.replace(',','').astype(float) 
ge_15['Other'] = ge_15['Votes'] - ge_15.fillna(0)['Con'] - ge_15.fillna(0)['Lab'] - ge_15.fillna(0)['LD'] - ge_15.fillna(0)['Grn'] - ge_15.fillna(0)['UKIP'] - ge_15.fillna(0)['SNP'] - ge_15.fillna(0)['PC']
ge_15["Don't vote"] = ge_15['Electorate'] - ge_15['Votes']
ge_15_region = ge_15[['Region', 'Electorate', 'Votes', 'Con', 'Lab', 'LD', 'Grn', 'UKIP', 'SNP', 'PC', 'Other', "Don't vote"]]
ge_15_region = ge_15_region.groupby('Region').sum()
ge_15_region = ge_15_region.drop(["SNP"], 1)
ge_15_region = ge_15_region.drop(["Northern Ireland", "Scotland"], 0)
ge_15_region = ge_15_region.rename(index={'East': 'East of England', 'Yorkshire and The Humber': 'Yorkshire and the Humber'})
ge_15_region_share = ge_15_region.div(ge_15_region['Votes'], axis=0)
ge_15_region_share['main'] = ge_15_region_share['Con'] + ge_15_region_share['Lab']

ge_15_region_share = ge_15_region_share.rename(index=region_categories)
ge_15_region_share = ge_15_region_share.rename(columns=party_categories)

"""Step 3: Load 1988 census data to enable poststratification."""
# this is a full joint distribution; if only marginal distributions are available then use raking
census_11 = pd.read_csv("/Users/davidminarsch/Desktop/PythonMLM/Election_Ex_Britain/2011_census/four_way_joint_distribution/four_way_joint_distribution_str.csv")
# age: categorical variable
# sex: indicator variable
# ethnicity: categorical variable
# region: categorical variable
# N: size of population in this cell

census_11_numeric = copy.deepcopy(census_11)
census_11_numeric["sex"] = census_11_numeric["sex"].apply(lambda x: sex_categories[x])
census_11_numeric["age"] = census_11_numeric["age"].apply(lambda x: age_categories[x])
census_11_numeric["ethnicity"] = census_11_numeric["ethnicity"].apply(lambda x: ethnicity_categories[x])
census_11_numeric["region"] = census_11_numeric["region"].apply(lambda x: region_categories[x])

#Test compatibility:
sorted(census_11.region.unique().tolist()) == sorted(polls.region.unique().tolist())
sorted(census_11.age.unique().tolist()) == sorted(polls.age.unique().tolist())
sorted(census_11.sex.unique().tolist()) == sorted(polls.sex.unique().tolist())
sorted(census_11.ethnicity.unique().tolist()) == sorted(polls.ethnicity.unique().tolist())
sorted(ge_10_region.index.tolist()) == sorted(ge_15_region.index.tolist())
sorted(ge_10_region.index.tolist()) == sorted(polls.region.unique().tolist())
#in ge_10_region.columns.tolist()) == sorted(polls.vote.unique().tolist())


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
  int<lower=0> n_region;
  int<lower=0> n_sex;
  int<lower=0> n_age;
  int<lower=0> n_ethnicity;
  vector[N] region_v_prev;
  int<lower=0,upper=n_region> region[N];
  int<lower=0,upper=n_sex> sex[N];
  int<lower=0,upper=n_age> age[N];
  int<lower=0,upper=n_ethnicity> ethnicity[N];
  int<lower=0,upper=1> y[N];
} 
parameters {
  vector[2] alpha;
  vector[n_region] a;
  vector[n_sex] b;
  vector[n_age] c;
  vector[n_ethnicity] d;
  real<lower=0,upper=100> sigma_a;
  real<lower=0,upper=100> sigma_b;
  real<lower=0,upper=100> sigma_c;
  real<lower=0,upper=100> sigma_d;
  real<lower=0> mu;
  real<lower=0,upper=100> sigma_0;
}
transformed parameters {
  vector[N] y_hat;
  for (i in 1:N)
    y_hat[i] = alpha[1] + alpha[2] * region_v_prev[region[i]] + a[region[i]] + b[sex[i]] + c[age[i]] + d[ethnicity[i]];
} 
model {
  a ~ normal (0, sigma_a);
  b ~ normal (0, sigma_b);
  c ~ normal (0, sigma_c);
  d ~ normal (0, sigma_d);
  alpha ~ normal(0, 100);
  sigma_a ~ scaled_inv_chi_square(mu,sigma_0);
  sigma_b ~ scaled_inv_chi_square(mu,sigma_0);
  sigma_c ~ scaled_inv_chi_square(mu,sigma_0);
  sigma_d ~ scaled_inv_chi_square(mu,sigma_0);
  mu ~ uniform(0, 100);
  sigma_0 ~ uniform(0, 100);
  y ~ bernoulli_logit(y_hat);
}
"""

# Model parameters and data:
model_1_data_dict = {'N': n, 'n_region': n_region, 'n_age': n_age, 'n_sex': n_sex, 'n_age': n_ethnicity,
  'region': polls_numeric.region, 'age': polls_numeric.age, 'sex': polls_numeric.sex, 'ethnicity': polls_numeric.ethnicity,
  'region_v_prev': ge_10_region_share['main'], 'y': polls_numeric.main}

# Fitting the model:
n_chains = 2
n_iter = 1000
#full_model_fit = pystan.stan(model_code=full_model, data=full_model_data_dict, iter=n_iter, chains=2)
sm = StanModel(model_code=model_1)
#with open('model_1.pkl', 'wb') as f:
#    pickle.dump(sm, f)
#sm = pickle.load(open('model_1.pkl', 'rb'))
model_1_fit = sm.sampling(data=model_1_data_dict, iter=n_iter, chains=n_chains)

import pdb; pdb.set_trace()

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