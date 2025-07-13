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
df = pd.read_csv('data/edstays.csv.gz', compression='gzip', header=0, sep=',', quotechar='"')
patientdf = pd.read_csv('data/patients copy.csv.gz', compression='gzip', header=0, sep=',', quotechar='"')
admdf = pd.read_csv('data/admissions copy 2.csv.gz', compression='gzip', header=0, sep=',', quotechar='"')
ddf = pd.read_csv('data/diagnosis.csv.gz', compression='gzip', header=0, sep=',', quotechar='"')
recdf = pd.read_csv('data/record_list copy.csv', dtype=dtype_spec)
ddf = pd.read_csv('data/triage.csv.gz', compression='gzip', header=0, sep=',', quotechar='"')
recdf = pd.read_csv('data/vitalsign.csv.gz', dtype=dtype_spec)
df = pd.merge(df, ECG_df, on='subject_id')
print(len(df))
df = df[(df['ecg_time'] >= df['intime']) & (df['ecg_time'] <= df['outtime'])]
df['outtime'] = pd.to_datetime(df['outtime'])
df['intime'] = pd.to_datetime(df['intime'])
df['ecg_time'] = pd.to_datetime(df['ecg_time'])
df['ECG_LoS'] = df['outtime'] - df['ecg_time']
df['LoS'] = df['outtime'] - df['intime']
df = pd.get_dummies(df, columns=['report_0', 'report_1', 'report_2', 'report_3', 'report_4', 'report_5', 'report_6', 'report_7', 'report_8','report_9','report_10','report_11','report_12','report_13','report_14','report_15','report_16','report_17'])
recdf['charttime'] = pd.to_datetime(recdf['charttime'])
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






print(f'Number of rows: {len(df)}')
print(f'Number of columns: {len(df.columns)}')
print(f'Number of sub: {len(df['subject_id'].unique())}')
print(f'Number of stay: {len(df['stay_id'].unique())}')


ddf.drop(columns=['subject_id'], inplace=True)
df = pd.merge(df, ddf, on='stay_id')


display(df.head(10))
print(f'Number of rows: {len(df)}')
print(f'Number of columns: {len(df.columns)}')
print(f'Number of sub: {len(df['subject_id'].unique())}')
print(f'Number of stay: {len(df['stay_id'].unique())}')
recdf.drop(columns=[
   'subject_id'
], inplace=True)


print(f'Number of rows: {len(df)}')
print(f'Number of columns: {len(df.columns)}')
print(f'Number of sub: {len(df['subject_id'].unique())}')
print(f'Number of stay: {len(df['stay_id'].unique())}')
display(df.head(10))


print(f'Number of rows: {len(df)}')
print(f'Number of columns: {len(df.columns)}')
print(f'Number of sub: {len(df["subject_id"].unique())}')
print(f'Number of stay: {len(df["stay_id"].unique())}')


display(df.head(10))

df.to_csv('df1', index=False)





import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
pd.set_option('display.max_rows', None) 
pd.set_option('display.max_columns', None) 
pd.set_option('display.width', None) 
df = pd.read_csv('df1')
df.columns = df.columns.str.rstrip('.')
duplicate_columns = df.columns[df.columns.duplicated()]
for col in duplicate_columns:
   col_mask = df.filter(like=col).any(axis=1)  # Combine True values across duplicate columns
   df[col] = col_mask


df = df.loc[:, ~df.columns.duplicated()]


keywords = ['chest pain', 'chest', 'dyspnea', 'cardiac', 'c/p', 'cp', 'syncope', 'presyncope']
pattern = '|'.join(keywords)


df = df[df['chiefcomplaint'].str.contains(pattern, case=False, na=False)]
print(len(df))
p_df = pd.read_csv('data/patients copy.csv.gz', compression='gzip', header=0, sep=',', quotechar='"')
p_df.drop(columns=['gender'], inplace=True)
df = pd.merge(df, p_df, on='subject_id')
df['intime'] = pd.to_datetime(df['intime']).dt.year
df['age'] = df['intime'] - df['anchor_year'] + df['anchor_age']
print(len(df))


df.drop(columns=[


   'hadm_id',


   'dod',
   'anchor_age',
   'anchor_year',
   'anchor_year_group'], inplace=True)




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




df = pd.get_dummies(df, columns=['gender', 'race', 'arrival_transport', 'disposition'])


df['a-v dissociation'] = df[['av dissociation', 'a-v dissociation']].apply(lambda x: x[0] or x[1], axis=1)




df.drop(columns=['av dissociation'], inplace=True)


df['ecg_los'] = pd.to_timedelta(df['ecg_los'])


# Extract total hours as float
df['ecg_los'] = df['ecg_los'].dt.total_seconds() / 3600






display(df.head(10))
df['los'] = pd.to_timedelta(df['los'])


# Extract total hours as float
df['los'] = df['los'].dt.total_seconds() / 3600
display(df.head(15))
# ――― count ECGs per stay ―――
counts_per_stay = df.groupby('stay_id').size()          # Series: stay_id → #ECGs


# ――― distribution: “how many stays had k ECGs?” ―――
dist = counts_per_stay.value_counts().sort_index()      # Series: k → #stays
print("ECG count distribution (k : #stays):")
print(dist)


# ――― summary stats ―――
mean_ecgs   = counts_per_stay.mean()
median_ecgs = counts_per_stay.median()
print(f"\nMean  #ECGs per stay: {mean_ecgs:.2f}")
print(f"Median #ECGs per stay: {median_ecgs}")


df.to_csv('final_df1', index=False)



# Load your DataFrame
import pandas as pd
import numpy as np
df1 = pd.read_csv('final_df1')
df_r = pd.read_csv('data/record_list copy.csv')


print(len(df1))
df = pd.merge(df1, df_r[['study_id','path']], on='study_id')
print(len(df))
df.drop(columns=['study_id', 'cart_id', 'filtering'], inplace=True)
display(df.head())
df['Chest Pain'] = df['chiefcomplaint'].str.contains(r'chest pain|chest|cardiac|CP|C/P', case=False, na=False)
df['Dyspnea'] = df['chiefcomplaint'].str.contains(r'dyspnea', case=False, na=False)
df['Presyncope'] = df['chiefcomplaint'].str.contains(r'presyncope', case=False, na=False)
df['Syncope'] = df['chiefcomplaint'].str.contains(r'syncope', case=False, na=False)


# Drop the 'pain' column
df.drop(columns=['pain',
                'chiefcomplaint',
                'pacer detection suspended due to external noise-review advised',
                'failure to sense and/or capture (?magnet)',
                '-------------------- pediatric ecg interpretation --------------------',
                'right and left arm electrode reversal, interpretation assumes no reversal',
                '--- possible measurement error ---',
                '--- pediatric criteria used ---',
                'age not entered, assumed to be  50 years old for purpose of ecg interpretation',
                'a-v dissociation with unclassified aberrant complexes',
                'arrival_transport_UNKNOWN',
                'arrival_transport_OTHER',
                'race_UNKNOWN',
                'race_PATIENT DECLINED TO ANSWER',
                'race_OTHER',
                'race_MULTIPLE RACE/ETHNICITY'
               
                ], inplace=True)


# Set display options
pd.set_option('display.max_rows', None)  # Display all rows
pd.set_option('display.max_columns', None)  # Display all columns
pd.set_option('display.width', None)  # No width limit






display(df.head(10))
df['pre_los'] = df['los'] - df['ecg_los']


display(df.head(10))


import numpy as np
import pandas as pd


# ----------------------------------------
# 0 .  Ensure chronological ordering first
# ----------------------------------------
# If you already have a proper datetime column 'ecg_time', use that.
# Otherwise rely on ecg_los (larger → earlier).  Here I use ecg_time.
df = df.sort_values(['stay_id', 'ecg_time'])          # earliest → latest


# -------------------------------------------------------
# 1 .  Per-stay reducer that returns a single formatted row
# -------------------------------------------------------
def collapse_stay(group):
   n = len(group)                     # how many ECGs in this stay
   # ---------- select final ECG row ----------
   if n >= 6:
       final_row = group.iloc[5]      # 6-th (0-based index)
       prior = group.iloc[:5]['path'] # paths of ECGs 1-5
   else:
       final_row = group.iloc[-1]     # very last tracing
       prior = group.iloc[:-1]['path']# everything before it


   # ---------- build path_1 … path_5 ----------
   prior_paths = prior.tolist() + [np.nan]*(5 - len(prior))
   path_cols   = {f'path_{i}': p for i, p in enumerate(prior_paths, 1)}


   # ---------- assemble output row ----------
   out = final_row.copy()
   out.rename({'path': 'final_ecg_path'}, inplace=True)
   out['n_ecg'] = n
   for k, v in path_cols.items():
       out[k] = v
   return out


# ----------------------------
# 2 .  Apply to every stay_id
# ----------------------------
result = (df.groupby('stay_id', group_keys=False)
           .apply(collapse_stay)
           .reset_index(drop=True))


print(result.head())


result.to_csv('sequential_ecgs.csv', index=False)



df = pd.read_csv('sequential_ecgs.csv')
print(df['subject_id'].nunique())
df_sorted = (
    df.sort_values(["subject_id", "n_ecg", "ecg_time"],
                   ascending=[True,       False,   False])
)

# ─────────────────────────────────────────────────────────────
# 2.  Drop the extra stays, keeping the first row per subject_id
# ─────────────────────────────────────────────────────────────
df_one_stay = df_sorted.drop_duplicates(subset="subject_id", keep="first")

# df_one_stay now has exactly one row per patient
print(f"Rows before: {len(df):,}  →  after: {len(df_one_stay):,}")
df_one_stay.to_csv('final_ecgs.csv', index = False)

# ── load the ECG-level table ────────────────────────────────────────────────
ecg_df = pd.read_csv("final_ecgs.csv", low_memory=False)

# ▸ disposition columns present in the CSV
disp_cols = [c for c in ecg_df.columns if c.startswith("disposition_")]

# ▸ keep rows with HOME or ADMIT *only*
mask_other = (ecg_df[disp_cols]
              .drop(columns=["disposition_HOME", "disposition_ADMITTED"],
                    errors="ignore") == 1).any(axis=1)

n_drop = mask_other.sum()
print(f"Dropping {n_drop} stays with non-HOME/ADMIT disposition …")

# ▸ filter & save
ecg_df_filtered = ecg_df.loc[~mask_other].reset_index(drop=True)
print(len(ecg_df_filtered))

ecg_df_filtered.to_csv("final_ecgs.csv", index=False)
