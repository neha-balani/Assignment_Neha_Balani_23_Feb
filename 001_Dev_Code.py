import pickle
import time
import warnings
from datetime import datetime

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
# from sklearn.impute import SimpleImputer
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import roc_auc_score, roc_curve, precision_score, recall_score
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

import Preprocess_00 as pp

warnings.filterwarnings(action='ignore', message='Mean of empty slice')

# disable chained assignments
pd.options.mode.chained_assignment = None
today = datetime.today()

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

# Read the datasets
Mortgage_dev = pd.read_csv('Mortgage.csv', na_values='?')
campaign_dev = pd.read_csv('Campaign.csv').add_suffix('_C')

# Preprocess datasets for merging
Mortgage_dev1 = pp.mort_data_preprocess(Mortgage_dev.copy())
campaign_dev1 = pp.camp_data_preprocess(campaign_dev.copy())

# # Create Model Data
campaign_dev1['target'] = np.where((campaign_dev1['created_account_C'] == 'Yes'), 1,
                                   (np.where(campaign_dev1['created_account_C'] == 'No', 0, -1))).astype('int')
customer_dev_df = campaign_dev1.merge(Mortgage_dev1, left_on=join_keys_C, right_on=join_keys_M, how='left')
customer_dev_df['title_match_flag'] = np.where(customer_dev_df['name_title'] == customer_dev_df['name_title_C'], 1, 0)
customer_dev_df['age_diff'] = np.abs(customer_dev_df.age - customer_dev_df.age_C)
customer_dev_df.sort_values(by=campaign_sort_keys, ascending=[True, True, True, True, False], inplace=True)
customer_dev_df = customer_dev_df.drop_duplicates(subset=campaign_keys)
customer_dev_df = customer_dev_df[(~customer_dev_df.duplicated(subset=mortgage_keys)) |
                                  ((customer_dev_df['full_name'].isnull())
                                   & (customer_dev_df['dob'].isnull()))]

# customer_dev_df = customer_dev_df.drop_duplicates(subset=mortgage_keys, keep=False)
customer_dev_df = customer_dev_df[customer_dev_df.created_account_C.notna()]

# Feature Engineering
customer_dev_df1 = pp.variables_creation(customer_dev_df)

# Data Aggregation
y = customer_dev_df1.loc[:, 'target']
X = pp.outlier_iqr(customer_dev_df1[cat_vars + cont_vars])

X_train0, X_test0, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

# Missing Imputation -No longer using and relying on XGBOOST SPARSITY MATRIX
# median_imputer = SimpleImputer(strategy='median')
# X_train0.loc[:, cont_vars] = median_imputer.fit_transform(X_train0.loc[:, cont_vars])
# X_test0.loc[:, cont_vars] = median_imputer.transform(X_test0.loc[:, cont_vars])
#
# mode_imputer = SimpleImputer(strategy='most_frequent')
# X_train0.loc[:, cat_vars] = mode_imputer.fit_transform(X_train0.loc[:, cat_vars].copy())
# X_test0.loc[:, cat_vars] = mode_imputer.transform(X_test0.loc[:, cat_vars].copy())

# Dummy Creation
ohe = OneHotEncoder(handle_unknown='ignore', sparse=False)
cat_ohe = ohe.fit_transform(X_train0[cat_vars])
ohe_X_train0 = pd.DataFrame(cat_ohe, columns=ohe.get_feature_names(input_features=cat_vars), index=X_train0.index)
X_train = pd.concat([X_train0, ohe_X_train0], axis=1).drop(columns=cat_vars, axis=1)

cat_ohe_new = ohe.transform(X_test0[cat_vars])
ohe_df_Test = pd.DataFrame(cat_ohe_new, columns=ohe.get_feature_names(input_features=cat_vars), index=X_test0.index)
X_test = pd.concat([X_test0, ohe_df_Test], axis=1).drop(columns=cat_vars, axis=1)

# Model Training and Evaluation

xgb = XGBClassifier(base_score=y_train.mean(),
                    use_label_encoder=False,
                    eval_metric='auc',
                    scale_pos_weight=50)

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=143)

param_grid = {"n_estimators": [150, 250, 300],
              #              "max_depth" : [2, 5, 8],
              #              "min_child_weight" : [1, 2, 5],
              #              "gamma" : [0, 0.1],
              #              "lambda" : [1,  4],
              #              "max_delta_step" : [0, 1, 2],
              "seed": [4],
              "objective": ["binary:logistic"]
              }
grid = GridSearchCV(xgb, param_grid=param_grid, cv=kfold)

start = time.time()
model = grid.fit(X_train, y_train)
end = time.time()
print('Time taken in seconds:', end - start)

train_pred = model.predict(X_train)
test_pred = model.predict(X_test)
test_probs = [i[1] for i in (model.predict_proba(X_test))]

print('Precision is:', precision_score(y_test, test_pred).round(2))
print('Recall is:', recall_score(y_test, test_pred).round(2))
print('Train Accuracy is:', model.score(X_train, y_train).round(2))
print('Test_accuracy is:', model.score(X_test, y_test).round(2))

fpr, tpr, thresholds = roc_curve(y_test, test_probs)

plt.title('ROC curve')
plt.xlabel('FPR (Precision)')
plt.ylabel('TPR (Recall)')

plt.plot(fpr, tpr)
plt.plot((0, 1), ls='dashed', color='black')
plt.show()

auc = roc_auc_score(y_test, test_pred)
print('Model AUC is : %.2f' % auc)

plot_confusion_matrix(model, X_test, y_test, cmap='Blues')
plt.show()

features = model.best_estimator_.feature_importances_
Var_imp_df = pd.DataFrame(X_train.columns)
Var_imp_df.columns = ['feature']
Var_imp_df['importance'] = (features * 100)
Var_imp_df.sort_values(by='importance', ascending=False, inplace=True)

feat_importances = pd.Series(model.best_estimator_.feature_importances_, index=X_train.columns)
feat_importances.nlargest(10).plot(kind='barh')

shap_values = shap.TreeExplainer(model.best_estimator_).shap_values(X_test)  # to explain every prediction,
shap.summary_plot(shap_values, X_test)  # then call  summary to plot these explanations
# shap.summary_plot(shap_values, X_test, plot_type="bar") #then call  summary to plot these explanations

# Dump Imputers and Ohe Creator

# mode_imputer_file = "mode_imputer.save"
# joblib.dump(mode_imputer, mode_imputer_file)
#
# median_imputer_file = "median_imputer.save"
# joblib.dump(median_imputer, median_imputer_file)

ohe_file = "ohe_creator.save"
joblib.dump(ohe, ohe_file)

file_name = 'XGboost_Campaign_model.pkl'
pickle.dump(grid, open(file_name, 'wb'))
