import pandas as pd
import numpy as np
import streamlit as st
from typing import Dict
from joblib import load


model = load("../../models/RF_hyperparameter_accuracy.joblib")

#Caching the model for faster loading
@st.cache
# list of inputs are too long so i use *args instead
def predict(*args):
    list_variables = list(args)
    data_input = pd.DataFrame([list_variables],columns= columns_list)
    data_predict = feature_process(data_input)

    prediction = model.predict(data_predict)
    return prediction

def feature_process(data: pd.DataFrame) -> pd.DataFrame:

    """Process raw data into useful files for model."""
    cols_to_drop = ['encounter_id',
                      'patient_nbr',
                      'examide',
                      'citoglipton',
                      'glimepiride-pioglitazone',
                      'weight',
                      'payer_code',
                      'diag_3',
                      'gender'
                      ]
    medication = ['metformin', 'repaglinide', 'nateglinide', 
            'chlorpropamide', 'glimepiride', 'glipizide', 
            'glyburide', 'pioglitazone', 'rosiglitazone', 
            'acarbose', 'miglitol', 'insulin', 
            'glyburide-metformin', 'tolazamide', 
            'metformin-pioglitazone','metformin-rosiglitazone',
            'glipizide-metformin', 'troglitazone', 'tolbutamide',
            'acetohexamide']

    med_specialty = ['Unknown', 'InternalMedicine', 'Family/GeneralPractice',
                     'Cardiology', 'Surgery-General', 'Orthopedics', 'Gastroenterology',
                     'Nephrology', 'Orthopedics-Reconstructive',
                     'Surgery-Cardiovascular/Thoracic', 'Pulmonology', 'Psychiatry',
                     'Emergency/Trauma', 'Surgery-Neuro', 'ObstetricsandGynecology',
                     'Urology', 'Surgery-Vascular', 'Radiologist']            

    cat_cols = ["admission_type_id", 
                "discharge_disposition_id",
                "admission_source_id"]                          

    process_data = (data
                    .pipe(replace_missing_values, replace_values='?')
                    .pipe(filter_cols_values, filter_col='discharge_disposition_id',
                          filter_values=[11, 13, 14, 19, 20])#test
                    .pipe(filter_cols_values, filter_col='diag_1', filter_values=[np.nan])#test
                    .pipe(fill_na_with_col, fill_col='diag_2', fill_col_from='diag_3')#test
                    .pipe(fill_na_with_col, fill_col='diag_2', fill_col_from='diag_1')#test
                    .pipe(fill_na_with_string, fill_col='medical_specialty', fill_string='Unknown')#test
                    .pipe(fill_na_with_string, fill_col='race', fill_string='Caucasian')#test
                    .pipe(encoding_columns)
                    .pipe(medication_changes,keys = medication)
                    .pipe(medication_encoding, keys = medication)
                    .pipe(diagnose_encoding)
                    .pipe(process_medical_specialty, keys = med_specialty)
                    .pipe(to_categorical, categorical_cols = cat_cols)
                    .pipe(drop_cols, drop_cols = cols_to_drop)
                    .pipe(encode_categorical)
                    .pipe(add_missing_cols)
                    )
    return process_data

def add_missing_cols(data: pd.DataFrame) -> pd.DataFrame:
    trains_cols = [
                'admission_type_id',
                'discharge_disposition_id',
                'admission_source_id',
                'time_in_hospital',
                'num_lab_procedures',
                'num_procedures',
                'num_medications',
                'number_outpatient',
                'number_emergency',
                'number_inpatient',
                'number_diagnoses',
                'max_glu_serum',
                'A1Cresult',
                'metformin',
                'repaglinide',
                'nateglinide',
                'chlorpropamide',
                'glimepiride',
                'acetohexamide',
                'glipizide',
                'glyburide',
                'tolbutamide',
                'pioglitazone',
                'rosiglitazone',
                'acarbose',
                'miglitol',
                'troglitazone',
                'tolazamide',
                'insulin',
                'glyburide-metformin',
                'glipizide-metformin',
                'metformin-rosiglitazone',
                'metformin-pioglitazone',
                'change',
                'diabetesMed',
                'numchange',
                'race_Asian',
                'race_Caucasian',
                'race_Hispanic',
                'race_Other',
                'age_[10-20)',
                'age_[20-30)',
                'age_[30-40)',
                'age_[40-50)',
                'age_[50-60)',
                'age_[60-70)',
                'age_[70-80)',
                'age_[80-90)',
                'age_[90-100)',
                'medical_specialty_Emergency/Trauma',
                'medical_specialty_Family/GeneralPractice',
                'medical_specialty_Gastroenterology',
                'medical_specialty_InternalMedicine',
                'medical_specialty_Nephrology',
                'medical_specialty_ObstetricsandGynecology',
                'medical_specialty_Orthopedics',
                'medical_specialty_Orthopedics-Reconstructive',
                'medical_specialty_Other',
                'medical_specialty_Psychiatry',
                'medical_specialty_Pulmonology',
                'medical_specialty_Radiologist',
                'medical_specialty_Surgery-Cardiovascular/Thoracic',
                'medical_specialty_Surgery-General',
                'medical_specialty_Surgery-Neuro',
                'medical_specialty_Surgery-Vascular',
                'medical_specialty_Unknown',
                'medical_specialty_Urology',
                'diag_1_Diabetes',
                'diag_1_Digestive',
                'diag_1_Genitourinary',
                'diag_1_Injury',
                'diag_1_Muscoloskeletal',
                'diag_1_Neoplasms',
                'diag_1_Others',
                'diag_1_Respiratory',
                'diag_2_Diabetes',
                'diag_2_Digestive',
                'diag_2_Genitourinary',
                'diag_2_Injury',
                'diag_2_Muscoloskeletal',
                'diag_2_Neoplasms',
                'diag_2_Others',
                'diag_2_Respiratory'
            ]
    missing_cols = set( trains_cols ) - set( data.columns )
    # Add a missing column in test set with default value equal to 0
    for c in missing_cols:
        data[c] = 0
    # Ensure the order of column in the test set is in the same order than in train set
    data = data[trains_cols]
    return data     