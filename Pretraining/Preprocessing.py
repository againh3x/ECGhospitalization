'''This file creates the full pretraining_ecgs.csv based on the MIMIC-IV-ECG database that will later be used to train a ResNet. 
The code creates the full cohort while merging and cleaning columns for pretraining.'''

import pandas as pd
from datetime import datetime, timedelta
pd.set_option('display.max_rows', None) 
pd.set_option('display.max_columns', None) 
pd.set_option('display.width', None) 
dtype_spec = {
   'report_17': 'str', 
   'report_12': 'str',
   'report_13': 'str',
   'report_14': 'str',
   'report_15': 'str',
   'report_16': 'str'
}




ECG_df = pd.read_csv('data/machine_measurements copy.csv', dtype=dtype_spec)
recdf = pd.read_csv('data/record_list copy.csv', dtype=dtype_spec)
df = pd.merge(ECG_df, recdf[['study_id', 'path']])

df = pd.get_dummies(df, columns=['report_0', 'report_1', 'report_2', 'report_3', 'report_4', 'report_5', 'report_6', 'report_7', 'report_8','report_9','report_10','report_11','report_12','report_13','report_14','report_15','report_16','report_17'])
def rename_columns(column_name):
   if column_name.startswith("report_"):
       return column_name.split("_", 2)[-1]  # Split by underscore and get the last part
   return column_name


# Apply the function to all column names
df.columns = [rename_columns(col) for col in df.columns]
df.columns = [col.lower() for col in df.columns]


# Identify duplicate columns
duplicate_columns = df.columns[df.columns.duplicated()]


# Merge duplicate columns by combining their True values
for col in duplicate_columns:
   col_mask = df.filter(like=col).any(axis=1)  # Combine True values across duplicate columns
   df[col] = col_mask


# Drop the original duplicate columns
df = df.loc[:, ~df.columns.duplicated()]




display(df.head())
df.drop(columns=['subject_id', 'study_id', 'cart_id', 'ecg_time', 'bandwidth', 'filtering'], inplace=True)
df.columns = df.columns.str.rstrip('.')
duplicate_columns = df.columns[df.columns.duplicated()]
for col in duplicate_columns:
   col_mask = df.filter(like=col).any(axis=1)  # Combine True values across duplicate columns
   df[col] = col_mask


df = df.loc[:, ~df.columns.duplicated()]


def remove_summary_prefix(df):
   # Rename columns by removing the "summary: " prefix
   df.columns = [col.replace("summary: ", "") for col in df.columns]
   return df


df = remove_summary_prefix(df)


duplicate_columns1 = df.columns[df.columns.duplicated()]
for col in duplicate_columns1:
   col_mask = df.filter(like=col).any(axis=1)  # Combine True values across duplicate columns
   df[col] = col_mask


df = df.loc[:, ~df.columns.duplicated()]
df['a-v dissociation'] = df[['av dissociation', 'a-v dissociation']].apply(lambda x: x[0] or x[1], axis=1)




df.drop(columns=['av dissociation'], inplace=True)
print(len(df))
print(len(df.columns))
df.drop(columns=[
                'pacer detection suspended due to external noise-review advised',
                'failure to sense and/or capture (?magnet)',
                '-------------------- pediatric ecg interpretation --------------------',
                'right and left arm electrode reversal, interpretation assumes no reversal',
                '--- possible measurement error ---',
                '--- pediatric criteria used ---',
                'age not entered, assumed to be  50 years old for purpose of ecg interpretation',
                'a-v dissociation with unclassified aberrant complexes',
               
                ], inplace=True)


pd.set_option('display.max_rows', None)  # Display all rows
pd.set_option('display.max_columns', None)  # Display all columns
pd.set_option('display.width', None)  # No width limit
import numpy as np
# Identify boolean columns
boolean_columns = [col for col in df.columns if df[col].dtype == 'bool']


# List to store columns to drop
columns_to_drop = []


# Count True values and identify columns with zero True values
for column in boolean_columns:
   true_count = df[column].sum()  # Count number of True values
   if true_count == 0:
       columns_to_drop.append(column)


# Drop columns with zero True values
df.drop(columns=columns_to_drop, inplace=True)


# Display the updated DataFrame




columns_to_drop = [col for col in df.columns if df[col].eq(True).sum() == 1]
# Drop those columns from the DataFrame
df.drop(columns=columns_to_drop, inplace=True)
columns_to_check = ['--- warning: data quality may affect interpretation ---', 'all 12 leads are missing', '--- suspect arm lead reversal - only avf, v1-v6 analyzed ---','--- recording unsuitable for analysis - please repeat ---', '--- suspect limb lead reversal - only v1-v6 analyzed ---', 'poor quality data, interpretation may be affected']


#Drop rows where either of the specified columns is True
df = df[~df[columns_to_check].any(axis=1)]


# Drop the specified columns
df = df.drop(columns=columns_to_check)
df = df.loc[:, ~df.columns.str.contains('lead\(s\)', case=False, na=False)]


keywords1 = ['complete av', 'third']
pattern = '|'.join(keywords1)
columns_to_combine1 = df.filter(regex=pattern).columns
df['3_degree_a-v'] = df[columns_to_combine1].any(axis=1)


keywords2 = ['second deg', '2:1', '3:1', '4:1', 'high degree', '2nd degree', '2:1 a-v block']
pattern = '|'.join(keywords2)
columns_to_combine2 = df.filter(regex=pattern).columns
df['2_degree_a-v'] = df[columns_to_combine2].any(axis=1)


keywords3 = ['1st degree', 'first degree', '- first degree a-v block', '- borderline first degree a-v block']
pattern = '|'.join(keywords3)
columns_to_combine3 = df.filter(regex=pattern).columns
df['1_degree_a-v'] = df[columns_to_combine3].any(axis=1)




columns_to_combine4 = df.filter(like='aberrantly conducted supraventricular complexes').columns
df['aberrantly conducted supraventricular complexes'] = df[columns_to_combine4].any(axis=1)


keywords6 = ['pvc', 'ventricular premature complex', 'multiple premature complexes, vent & supraven', 'premature ventricular complex', 'premature ventricular contractions', 'ventricular couplets', 'bigeminal pvcs', '- frequent premature ventricular contractions', '- premature ventricular contractions', '- ventricular couplets']
pattern = '|'.join(keywords6)
columns_to_combine4 = df.filter(regex=pattern).columns
df['pvc(s)'] = df[columns_to_combine4].any(axis=1)


keywords6 = ['pacs', 'pac(s)', 'atrial premature complex', 'atrial premature complexes', 'premature atrial complex', 'premature atrial complexes', 'premature atrial contractions', 'supraventricular extrasystoles', '- premature atrial contractions', '- supraventricular extrasystoles']
pattern = '|'.join(keywords6)
columns_to_combine4 = df.filter(regex=pattern).columns
df['pac(s)'] = df[columns_to_combine4].any(axis=1)


columns_to_combine5 = df.filter(like='pace').columns
df['paced'] = df[columns_to_combine5].any(axis=1)


keywords6 = ['v-rate', 'rapid ventricular response', 'with rapid ventricular response']
pattern = '|'.join(keywords6)
columns_to_combine6 = df.filter(regex=pattern).columns
df['RVR'] = df[columns_to_combine6].any(axis=1)


keywords6 = ['slow ventricular response', 'with slow ventricular response']
pattern = '|'.join(keywords6)
columns_to_combine6 = df.filter(regex=pattern).columns
df['SVR'] = df[columns_to_combine6].any(axis=1)


keywords6 = ['ivcd', 'iv conduction defect', 'intraventricular conduction delay']
pattern = '|'.join(keywords6)
columns_to_combine6 = df.filter(regex=pattern).columns
df['IVCD'] = df[columns_to_combine6].any(axis=1)


keywords6 = ['early repolarization', 'possible early repolarization']
pattern = '|'.join(keywords6)
columns_to_combine6 = df.filter(regex=pattern).columns
df['early repolarization'] = df[columns_to_combine6].any(axis=1)




keywords6 = ['lbbb', 'left bundle branch block']
pattern = '|'.join(keywords6)
columns_to_combine6 = df.filter(regex=pattern).columns
df['LBBB'] = df[columns_to_combine6].any(axis=1)


keywords6 = ['rbbb', 'right bundle branch block']
pattern = '|'.join(keywords6)
columns_to_combine6 = df.filter(regex=pattern).columns
df['RBBB'] = df[columns_to_combine6].any(axis=1)


keywords6 = ['lafb', 'left anterior fascicular block']
pattern = '|'.join(keywords6)
columns_to_combine6 = df.filter(regex=pattern).columns
df['LAFB'] = df[columns_to_combine6].any(axis=1)


keywords6 = ['dextrocardia', '--- suggests dextrocardia ---']
pattern = '|'.join(keywords6)
columns_to_combine6 = df.filter(regex=pattern).columns
df['dextrocardia'] = df[columns_to_combine6].any(axis=1)


keywords6 = ['lpfb', 'left posterior fascicular block']
pattern = '|'.join(keywords6)
columns_to_combine6 = df.filter(regex=pattern).columns
df['LPFB'] = df[columns_to_combine6].any(axis=1)


keywords6 = ['lad', 'left axis deviation', 'leftward axis']
pattern = '|'.join(keywords6)
columns_to_combine6 = df.filter(regex=pattern).columns
df['Left axis dev'] = df[columns_to_combine6].any(axis=1)


keywords6 = ['right axis deviation', 'rightward axis']
pattern = '|'.join(keywords6)
columns_to_combine6 = df.filter(regex=pattern).columns
df['Right axis dev'] = df[columns_to_combine6].any(axis=1)


columns_to_combine7 = df.filter(like='prolonged pr interval').columns
df['long pr interval'] = df[columns_to_combine7].any(axis=1)




keywords6 = ['pace', 'pacing']
pattern = '|'.join(keywords6)
columns_to_combine6 = df.filter(regex=pattern).columns
df['paced'] = df[columns_to_combine6].any(axis=1)


columns_to_combine7 = df.filter(like='idioventricular rhythm').columns
df['idioventricular_rhythm'] = df[columns_to_combine7].any(axis=1)


columns_to_combine8 = df.filter(like='junctional rhythm').columns
df['junctional_rhythm'] = df[columns_to_combine8].any(axis=1)


columns_to_combine8 = df.filter(like='uncontrolled ventricular response').columns
df['UVR'] = df[columns_to_combine8].any(axis=1)


columns_to_combine8 = df.filter(like='sinus arrhythmia').columns
df['sinus_arrh'] = df[columns_to_combine8].any(axis=1)


keywords6 = ['lae', 'left atrial enlargement', 'biatrial enlargement']
pattern = '|'.join(keywords6)
columns_to_combine6 = df.filter(regex=pattern).columns
df['LA enlargement'] = df[columns_to_combine6].any(axis=1)


keywords6 = ['rae', 'right atrial enlargement', 'biatrial enlargement']
pattern = '|'.join(keywords6)
columns_to_combine6 = df.filter(regex=pattern).columns
df['RA enlargement'] = df[columns_to_combine6].any(axis=1)


df['low voltage precordial'] = df['low voltage, precordial leads'] | df['low voltage, extremity and precordial leads'] | df['low qrs voltages in precordial leads'] | df['generalized low qrs voltages']
df['low voltage limb'] = df['low qrs voltages in limb leads'] | df['low voltage, extremity and precordial leads'] | df['low voltage, extremity leads'] | df['generalized low qrs voltages']


df['SV Tachycardia'] = df['probable supraventricular tachycardia'] | df['supraventricular tachycardia']
keywords = ['WHITE', 'ASIAN', 'BLACK', 'HISPANIC', 'ventricular hypertrophy', 'race_BLACK', 'depression', 'pericarditis', 'st elevation', 'infarct, old', 'infarct, acute', 'infarct - possibly acute', 'infarct - age indeterminate', 'infarct, age indeterminate', 'infarct - age undetermined', 'infarct, possibly acute', 'infarct, recent', 'ischemia', 'atrial fib', 'atrial flut', 'ectopic atrial r', 'ectopic atrial b', 'ectopic atrial t', 'sinus br', 'sinus rhy', 'sinus tac', ' infarct', 'lvh', 't wave changes', 'st changes', 'st-t changes', 't abnormalities', 't abnrm', 'left atrial abnorm', 'repol abnormality']  # Add more keywords as needed


columns_to_drop = [col for col in df.columns if ('pace' in col.lower() or 'pacing' in col.lower()) and col != 'paced']
df.drop(columns=columns_to_drop, inplace=True)
columns_to_drop = [col for col in df.columns if ('idioventricular rhythm' in col.lower())]
df.drop(columns=columns_to_drop, inplace=True)
columns_to_drop = [col for col in df.columns if ('junctional rhythm' in col.lower())]
df.drop(columns=columns_to_drop, inplace=True)
columns_to_drop = [col for col in df.columns if ('sinus arrhythmia' in col.lower())]
df.drop(columns=columns_to_drop, inplace=True)
columns_to_drop = [col for col in df.columns if ('atrial premature complex' in col.lower())]
df.drop(columns=columns_to_drop, inplace=True)
# Process each keyword
for keyword in keywords:
   # Find columns containing the current keyword
   columns_to_combine = df.filter(like=keyword).columns


   # Create a new column with combined True values
   df[keyword] = df[columns_to_combine].any(axis=1)


   # Drop the original columns that were combined
   df = df.drop(columns=columns_to_combine)






df = df.drop(columns=[col for col in df if col not in {'path'} and df[col].sum() < 10])


df['repol abnormality'] = df['repolarization abnormality, prob rate related'] | df['repol abnormality']
df['infarct - age undetermined'] = df['infarct, age indeterminate'] | df['infarct - age undetermined']
df['infarct, acute'] = df['infarct - possibly acute'] | df['infarct, possibly acute'] | df['infarct, acute'] | df[' infarct']
df['t abnormalities'] = df['t abnormalities'] | df['t abnrm']
df['ventricular hypertrophy'] = df['ventricular hypertrophy'] | df['lvh']
df['abnormal ecg'] = df['abnormal ecg'] | df['abnormal']
df['borderline ecg'] = df['borderline ecg'] | df['borderline']
df.drop(columns=[
   'av block, complete (third degree)',
   'undetermined rhythm',
   'unknown rhythm, irregular rate',
   'wide-qrs tachycardia',
   'abnormal ventricular conduction pathways',
   'biatrial enlargement',
   'borderline high qrs voltage - probable normal variant',
   'generalized low qrs voltages',
   'iv conduction defect',
   'ivcd, consider atypical lbbb',
   'ivcd, consider atypical rbbb',
   'incomplete lbbb',
   'incomplete rbbb',
   'incomplete rbbb and lafb',
   'incomplete left bundle branch block',
   'incomplete right bundle branch block',
   'indeterminate axis',
   'lad, consider left anterior fascicular block',
   'lae, consider biatrial enlargement',
   'left anterior fascicular block',
   'left atrial enlargement',
   'left axis deviation',
   'left bundle branch block',
   'left posterior fascicular block',
   'leftward axis',
   'low qrs voltages in limb leads',
   'low qrs voltages in precordial leads',
   'low voltage with right axis deviation',
   'low voltage, extremity and precordial leads',
   'low voltage, extremity leads',
   'low voltage, precordial leads',
   'multiform ventricular premature complexes',
   'multiple premature complexes, vent & supraven',
   'multiple ventricular premature complexes',
   'nonspecific ivcd with lad',
   'nonspecific intraventricular conduction delay',
   'paired ventricular premature complexes',
   'possible biatrial enlargement',
   'possible faulty v2 - omitted from analysis',
   'possible left anterior fascicular block',
   'possible left posterior fascicular block',
   'possible sequence error: v2,v3 omitted',
   'probable left atrial enlargement',
   'prominent p waves, nondiagnostic',
   'rbbb and lafb',
   'rbbb and lpfb',
   'rbbb with left anterior fascicular block',
   'rvh with secondary repolarization abnormality',
   'right atrial enlargement',
   'right axis deviation',
   'right bundle branch block',
   'rightward axis',
   's1,s2,s3 pattern',
   'severe right axis deviation',
   'borderline prolonged pr interval',
   'prolonged pr interval',
   'ventricular premature complex',
   'short qt interval',
   'anterior q waves, possibly due to ilbbb',
   'extensive ivcd',
   'lateral leads are also involved',
   'normal ecg based on available leads',
   'supraventricular tachycardia',
   'probable supraventricular tachycardia',
   'repolarization abnormality, prob rate related',
   'infarct, age indeterminate',
   'infarct, possibly acute',
   'infarct - possibly acute',
   ' infarct',
   't abnrm',
   'p_onset'
], inplace=True)
df['qrs_duration'] = df['qrs_end'] - df['qrs_onset']
df = df.applymap(lambda x: np.nan if isinstance(x, (int, float)) and x > 10000 else x)
print("Remaining columns:", len(df.columns))


display(df.head(10))

cols_to_keep = ['bandwidth', 'early repolarization', 'rr_interval', 'SVR', 'or aberrant ventricular conduction', 'p_end', 'qrs_onset', 'qrs_end', 't_end', 'p_axis', 'qrs_axis', 't_axis', 'ecg_los', 'a-v dissociation', 'dextrocardia', 'junctional tachycardia', 'predominant 3:1 av block', 'regular supraventricular rhythm', 'aberrant complex', 'aberrant conduction of sv complex(es)', 'abnormal r-wave progression, early transition', 'poor r wave progression - probable normal variant', 'possible right atrial abnormality', 'prolonged qt interval', 'st elev, probable normal early repol pattern', 'short pr interval', 'sinus pause', 'supraventricular bigeminy', 'ventricular bigeminy', 'ventricular tachycardia, unsustained', 'ventricular trigeminy', "rsr'(v1) - probable normal variant", 'abnormal ecg', 'borderline ecg', 'normal ecg', 'normal ecg except for rate', 'temperature', 'heartrate', 'o2sat', 'sbp', 'dbp', 'acuity', 'age', 'gender_F', 'gender_M', 'race_AMERICAN INDIAN/ALASKA NATIVE', 'race_NATIVE HAWAIIAN OR OTHER PACIFIC ISLANDER', 'race_PORTUGUESE', 'race_SOUTH AMERICAN', 'arrival_transport_AMBULANCE', 'arrival_transport_HELICOPTER', 'arrival_transport_WALK IN', 'path', 'Chest Pain', 'Dyspnea', 'Presyncope', 'Syncope', '3_degree_a-v', '2_degree_a-v', '1_degree_a-v', 'aberrantly conducted supraventricular complexes', 'pvc(s)', 'pac(s)', 'paced', 'RVR', 'IVCD', 'LBBB', 'RBBB', 'LAFB', 'LPFB', 'Left axis dev', 'Right axis dev', 'long pr interval', 'idioventricular_rhythm', 'junctional_rhythm', 'UVR', 'sinus_arrh', 'LA enlargement', 'RA enlargement', 'low voltage precordial', 'low voltage limb', 'SV Tachycardia', 'WHITE', 'ASIAN', 'BLACK', 'HISPANIC', 'ventricular hypertrophy', 'depression', 'pericarditis', 'st elevation', 'infarct, old', 'infarct, acute', 'infarct - age undetermined', 'infarct, recent', 'ischemia', 'atrial fib', 'atrial flut', 'ectopic atrial r', 'ectopic atrial b', 'ectopic atrial t', 'sinus br', 'sinus rhy', 'sinus tac', 'lvh', 't wave changes', 'st changes', 'st-t changes', 't abnormalities', 'left atrial abnorm', 'repol abnormality', 'qrs_duration']


dropped_cols = [col for col in df.columns if col not in cols_to_keep]


# Check for columns with more than 200 True values
for col in dropped_cols:
   if col in df.columns and df[col].dtype == bool:
       if df[col].sum() > 200:
           print(f"Column '{col}' has {df[col].sum()} True values.")


df = df[[col for col in cols_to_keep if col in df.columns]]
print(len(df.columns))
df.to_csv('pretraining_ecgs.csv', index=False)
