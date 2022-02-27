import warnings

import joblib
import numpy as np
import pandas as pd

import Preprocess_00 as pp

import pickle
import matplotlib.pyplot as plt

# disable chained assignments
pd.options.mode.chained_assignment = None
warnings.filterwarnings(action='ignore', message='Mean of empty slice')

# mode_imputer = joblib.load('mode_imputer.save')
# median_imputer = joblib.load('median_imputer.save')
ohe_creator = joblib.load('ohe_creator.save')

mortgage_keys = ['full_name', 'dob']
campaign_keys = ['first_name_C', 'last_name_C', 'company_email_C']
join_keys_M = ['first_name', 'last_name', 'Postcode_area']
join_keys_C = ['first_name_C', 'last_name_C', 'area_C']
campaign_sort_keys = ['first_name_C', 'last_name_C', 'area_C', 'age_diff', 'title_match_flag']
mortgage_sort_keys = ['first_name', 'last_name', 'Postcode_area', 'age_diff', 'title_match_flag']

cat_vars = ['sex', 'religion', 'race', 'native_country', 'workclass',
            'marital_status_C', 'relationship', 'married_flg1', 'interested_insurance_C',
            'area_C', 'area_group_C', 'Post_town', 'Postcode_area', 'native_UK',
            'edu_group_C', 'education_C', 'edu_grad_plus_fg', 'edu_pgrad_plus_fg',
            'job_title_imp', 'tenure_range1',
            'capital_gain_flg', 'capital_loss_flg', 'Currency_pound_fg', 'salary_frequency']

cont_vars = ['age_C', 'hours_per_week', 'demographic_characteristic', 'capital_gain', 'capital_loss',
             'occupation_level_C',
             'education_num_C', 'familiarity_FB_C', 'view_FB_C', 'time_curr_emp', 'Salary_calculated']

target = ['target']

Mortgage_prod = pd.read_csv('Mortgage.csv', na_values='?')
campaign_prod = pd.read_csv('Campaign.csv').add_suffix('_C')

Mortgage_prod1 = pp.mort_data_preprocess(Mortgage_prod.copy())
campaign_prod1 = pp.camp_data_preprocess(campaign_prod.copy())

customer_prod_df = Mortgage_prod1.merge(campaign_prod1, left_on=join_keys_M, right_on=join_keys_C, how='left')
customer_prod_df['title_match_flag'] = np.where(customer_prod_df['name_title'] == customer_prod_df['name_title_C'], 1,
                                                0)

customer_prod_df['age_diff'] = np.abs(customer_prod_df.age - customer_prod_df.age_C)
customer_prod_df.sort_values(by=mortgage_sort_keys, ascending=[True, True, True, True, False], inplace=True)

customer_prod_df = customer_prod_df.drop_duplicates(subset=mortgage_keys).copy()
customer_prod_df.drop(columns=['age_C'], inplace=True)
customer_prod_df.rename(columns={'age': 'age_C'}, inplace=True)

customer_prod_df = customer_prod_df[(~customer_prod_df.duplicated(subset=campaign_keys)) |
                                    ((customer_prod_df['first_name_C'].isnull()) &
                                     (customer_prod_df['last_name_C'].isnull()) &
                                     (customer_prod_df['company_email_C'].isnull()))]

# Feature Engineering

customer_prod_df1 = pp.variables_creation(customer_prod_df)

if 'created_account_C' in customer_prod_df1.columns:
    del customer_prod_df1['created_account_C']

# outlier treatment
customer_prod_df2 = pp.outlier_iqr(customer_prod_df1[cat_vars + cont_vars])
#
# # Missing Imputation
#
# customer_prod_df2.loc[:, cont_vars] = median_imputer.transform(customer_prod_df2.loc[:, cont_vars])
# customer_prod_df2.loc[:, cat_vars] = mode_imputer.transform(customer_prod_df2.loc[:, cat_vars].copy())
# assert customer_prod_df2.isna().sum().sum() == 0

# Dummy Creation

cat_ohe_new = ohe_creator.transform(customer_prod_df2[cat_vars])
ohe_df_Test = pd.DataFrame(cat_ohe_new, columns=ohe_creator.get_feature_names(input_features=cat_vars),
                           index=customer_prod_df2.index)
X_pred = pd.concat([customer_prod_df2, ohe_df_Test], axis=1).drop(columns=cat_vars, axis=1)

# Predict Customer Response

saved_model_file_name = 'XGboost_Campaign_model.pkl'
loaded_model_xgboost = pickle.load(open(saved_model_file_name, 'rb'))
y_pred = loaded_model_xgboost.predict(X_pred).round()
print(y_pred.sum(), ' Number of customers should be targeted to buy insurance as per our model')

pred_p = loaded_model_xgboost.predict_proba(X_pred)

len(y_pred), customer_prod_df.shape

customer_prod_df['predicted_response'] = y_pred
customer_prod_df['probability_response'] = [i[1] for i in pred_p]

x = customer_prod_df['probability_response'].max()
'{:f}'.format(x)

Campaign_customers_df = customer_prod_df[customer_prod_df['predicted_response'] == 1]

plt.hist(customer_prod_df['probability_response'], bins=10, )
plt.title('predicted probability distribution across all customers')
plt.ylabel('# of customers')
plt.xlabel('buying insurance -Probability')
plt.show()
