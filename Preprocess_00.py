import numpy as np
import pandas as pd

import re
import warnings
from datetime import datetime

import requests
from bs4 import BeautifulSoup
from nameparser import HumanName
from postcodes_uk import Postcode

warnings.filterwarnings(action='ignore', message='Mean of empty slice')

# disable chained assignments
pd.options.mode.chained_assignment = None
today = datetime.today()


# Variable Creation Functions

def town_outcode(town):
    """ Get Town area and outcode from
    postcode using postcodes_uk package"""
    area = Postcode.from_string(town).area
    outcode = Postcode.from_string(town).outward_code
    return area, outcode


# Scrape Post code details from wikipedia page
def scrape_post_code():
    """ Scrape post code reference table from wikipedia
    to extract area code from town for merging purposes"""

    url = 'https://en.wikipedia.org/wiki/List_of_postcode_districts_in_the_United_Kingdom'  # Create a URL object
    page = requests.get(url)  # Create object page
    soup = BeautifulSoup(page.text, 'lxml')  # Obtain page's information
    wiki_table = soup.find('table', {'class': 'wikitable sortable'})  # Obtain information from tag <table>
    wiki_table1 = pd.read_html(str(wiki_table))  # convert list to dataframe
    wiki_table02 = pd.DataFrame(wiki_table1[0])
    return wiki_table02


wiki_table2 = scrape_post_code()


def extract_parts(name):
    """ parse a name using name parser package
        into different part of name"""
    title, first, middle, last = (HumanName(name)).title, (HumanName(name)).first, (HumanName(name)).middle, (
        HumanName(name)).last
    return title, first, middle, last


def outlier_iqr(data):
    """create an outlier treatment code and replace
    outlier values with the lower and upper bounds """
    for col in data.columns:
        if data[col].dtype != object:
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            S = 1.5 * IQR
            LB = Q1 - S
            UB = Q3 + S
            data.loc[data[col] > UB, col] = UB
            data.loc[data[col] < LB, col] = LB
        else:
            break
    return data


def age_calculator(dob):
    """calculate age given the dob"""
    dob_date = pd.to_datetime(dob)
    age = today.year - dob_date.year - ((today.month, today.day) < (dob_date.month, dob_date.day))
    return age


def salary_fields(data):
    """scrape salary details from salary field"""
    data['salary_band'].fillna('', inplace=True)
    data['Salary_list'] = data['salary_band'].apply(lambda x: (re.sub("[^-0-9.,-]+", '', x).split('-')))
    data['Salary_raw'] = data['Salary_list'].apply(
        lambda x: np.nanmean(np.array([None if z == '' else z for z in x], dtype=float)))
    data['Salary_text'] = data['salary_band'].apply(lambda x: re.sub("[-0-9.,-]+", '', x))
    annual = 'yearly|year|per_year|per year'
    monthly = 'month'
    weekly = 'week|pw'
    data['Salary_multiplier'] = np.where(data['Salary_text'].str.contains(annual), 1,
                                         (np.where(data['Salary_text'].str.contains(monthly), 12,
                                                   (np.where(data['Salary_text'].str.contains(weekly), 52.14, 1)))))

    data['salary_frequency'] = np.where(data['Salary_text'].str.contains(annual), 'Annual',
                                        (np.where(data['Salary_text'].str.contains(monthly), 'Monthly',
                                                  (np.where(data['Salary_text'].str.contains(weekly), 'Weekly',
                                                            'Annual')))))

    data['Salary_calculated'] = data['Salary_multiplier'].fillna(1) * data['Salary_raw'].fillna(0)
    currency_string = 'annual|pw|yearly|year|week|pw|monthly|month|per week|per month|SOS|range|\t|" "'
    data['Currency'] = data['Salary_text'].str.replace(currency_string, '', regex=True)
    data['Currency'] = data['Currency'].str.strip()
    data['Currency_pound_fg'] = np.where(data['Currency'] == '£', 1, 0)

    return data['Salary_calculated'], data['salary_frequency'], data['Currency'], data['Currency_pound_fg']


# list of job titles which have high response rate
imp_job_titles = ['Animator', 'Associate Professor', 'Chief Technology Officer', 'Department Chair',
                  'Forensic psychologist', 'Head of Data Services', 'Investment banker, operational',
                  'Lead Data Scientist', 'Multimedia specialist', 'Osteopath', 'Personnel officer',
                  'President', 'Prime Minister', 'Scientist, research (medical)', 'Sub']


def variables_creation(data):
    """ Code to create multiple variables for modelling input"""
    high_fequency_areas = ['EH', 'SN', 'LS', '148', 'OX', 'BS', 'S', 'LE', 'SR', 'W' 'LN']
    data['area_group_C'] = np.where(data['area_C'].isin(high_fequency_areas), data['area_C'], 'Others')
    data['married_flg1'] = np.where(data['marital_status_C'].str.startswith('Married'), 1, 0)
    some_school = ['Preschool', 'SN', '1st-4th', '5th-6th', '9th', 'Prof-school', '7th-8th', '10th', '11th', 'W' 'LN']
    assoc = ['Assoc-voc', 'Assoc-acdm']
    data['edu_group_C'] = np.where(data['education_C'].isin(some_school), 'some_school', data['education_C'], )
    data['edu_group_C'] = np.where(data['edu_group_C'].isin(assoc), 'assoc', data['edu_group_C'], )
    data['edu_grad_plus_fg'] = np.where(data['education_C'].isin(['HS-grad', 'Masters', 'Doctorate']), 1, 0)
    data['edu_pgrad_plus_fg'] = np.where(data['education_C'].isin(['Masters', 'Doctorate']), 1, 0)
    data['job_title_imp'] = data['job_title_C'].isin(imp_job_titles).astype(int)  # response encoded list
    data['native_UK'] = (np.where(data['native_country'] == 'United Kingdom', 1, 0))
    data['months_with_employer1'] = (data['months_with_employer'] / 12).round(1)
    data['time_curr_emp'] = data[['years_with_employer', 'months_with_employer1']].sum(axis=1, min_count=1)
    bins = [0, 2, 5, 10, 65, 75]
    tenure_band = ['Lt2', '2_5', '5_7', '7_10', 'MT10']
    data['tenure_range1'] = pd.cut(data['time_curr_emp'], bins, labels=tenure_band, include_lowest=True)
    data['capital_gain_flg'] = np.where(data['capital_gain'] > 0, 1, 0)
    data['capital_loss_flg'] = np.where(data['capital_loss'] > 0, 1, 0)
    data['Salary_calculated'], data['salary_frequency'], data['Currency'], data['Currency_pound_fg'] = salary_fields(
        data)
    return data


# Campaign preprocessing
def camp_data_preprocess(data):
    """ add area and outcode details to campaign file using outcode function"""
    data['area_C'] = data['postcode_C'].apply(lambda x: town_outcode(x)[0])
    data['outer_code_C'] = data['postcode_C'].apply(lambda x: town_outcode(x)[1])
    return data


# Mortgage Preprocessing
def mort_data_preprocess(data):
    """parse names of mortgage data into title,
    first name, middle name and last name column"""
    data['name_title'], data['first_name'], data['middle_name'], data['last_name'] = zip(
        *data['full_name'].apply(extract_parts))
    data['dob'] = pd.to_datetime(data['dob'])
    data['age'] = data['dob'].apply(lambda x: age_calculator(x))
    data = data.merge(wiki_table2[['Post town', 'Postcode area']], left_on='town', right_on=['Post town'],
                      how='left').rename(columns={'Postcode area': 'Postcode_area', 'Post town': 'Post_town'})
    return data
