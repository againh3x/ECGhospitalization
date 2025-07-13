# Serial 12-Lead ECG–Based Deep-Learning Model for Real-Time Prediction of Hospital Admission in Emergency-Department Cardiac Presentations: Retrospective Cohort Study
Arda Altintepe

The scripts and models for this project are provided here in three directories. Many scripts take in MIMIC-IV raw or derived CSVs which are open-source but cannot be shared on a public GitHub repository. 

Preprocessing (inputs: MIMIC-IV raw csvs; ouput: 'final_ecgs.csv')

.main.py - The main preprocessing script. Takes in all raw MIMIC-IV files and establishes the full cohort of subjects with some corresponding triage and demographic features. Outputs 'final_ecgs.csv' to be used for further feature extraction and training.

EDmeds.py - Inputs final_ecgs.csv and adds 38 binary features corresponding to bins of medications administered in the ED before cutoff.

PMH_diagnoses.py - Inputs final_ecgs.csv and adds 253 bins corresponding to prior MIMIC-IV ICD diagnoses for each patient. 

PMH_meds.py - Inputs final_ecgs.csv and adds 7 broader medication bins corresponding to ETC codes of visible medications prescribed in the past year.

labs.py - Inputs final_ecgs.csv and adds 36 new features corresponding to 9 lab results visible before prediction cutoff. Each lab result contains columns for first value, peak value, abnormal, and missing. 

vitals.py - Inputs final_ecgs.csv and creates a new csv labeled 'vitals_long_cleaned.csv'. This csv contains rows for all vitals charted within the prediction cutoff corresponding to each stay in final_ecgs.csv. 'Pain' is simply imputed with '-1' for missing values and 2 flags for unable/sleeping and 1 flag for missing. 


Pretraining (inputs: MIMIC-IV raw csvs; output: 'pretrained_model.pth')
