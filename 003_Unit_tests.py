# check if development dataset and production dataset have same X inputs

import os

os.system('python Preprocess_00.py')
os.system('python 001_Dev_code.py')
os.system('python 002_Prod_code.py')

# check if raw files received have the required column names
assert campaign_prod.columns.values in campaign_dev.columns.values

# ### check if raw files received have the required column names
assert campaign_prod.columns.values in campaign_dev.columns.values

try:
    assert campaign_prod.columns.values in campaign_dev.columns.values
except:
    pass
    print('Campaign Prod Data dont not have all the required columns')

try:
    assert Mortgage_prod.columns.values in Mortgage_dev.columns.values
except:
    pass
    print('Mortgage Prod Data dont not have all the required columns')

# check if raw data has duplicates at unique key level
try:
    assert campaign_dev.shape[0] == \
           campaign_prod.drop_duplicates(subset=['first_name_C', 'last_name_C', 'company_email_C']).shape[0]
except:
    pass
    print('Campaign Prod Data received has duplicates at key level')

try:
    assert Mortgage_dev1.shape[0] == Mortgage_prod1.drop_duplicates(subset=['full_name', 'dob']).shape[0]
except:
    pass
    print('Mortgage Prod Data received has duplicates at key level')

# Ensure Prod data does not have customer response column when starting preprocessing
try:
    assert 'target1' not in customer_prod_df.columns
except:
    pass
    print('Prod Data already has response column')

# Ensure Prod and Dev data have same Columns and their dtypes
try:
    assert X_train.columns.values in X_pred.columns.values
except:
    pass
    print('Production data doesnt have all the required columns created for prediction')

# Ensure Production Data has no Missing values
try:
    assert X_pred.isna().sum().sum() == 0
except:
    pass
    print('Prod Data has Missing values')
