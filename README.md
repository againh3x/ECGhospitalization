# Serial 12-Lead ECG–Based Deep-Learning Model for Real-Time Prediction of Hospital Admission in Emergency-Department Cardiac Presentations: Retrospective Cohort Study  
Arda Altintepe

The scripts and models for this project are provided here in three directories. Many scripts take in MIMIC-IV raw or derived CSVs which are open-source but cannot be shared on a public GitHub repository.  

---

## Preprocessing (inputs: MIMIC-IV raw csvs; ouput: `final_ecgs.csv`)

- `.main.py` - The main preprocessing script. Takes in all raw MIMIC-IV files and establishes the full cohort of subjects with some corresponding triage and demographic features. Outputs `final_ecgs.csv` to be used for further feature extraction and training.  
- `EDmeds.py` - Inputs `final_ecgs.csv` and adds 38 binary features corresponding to bins of medications administered in the ED before cutoff.  
- `PMH_diagnoses.py` - Inputs `final_ecgs.csv` and adds 253 bins corresponding to prior MIMIC-IV ICD diagnoses for each patient.  
- `PMH_meds.py` - Inputs `final_ecgs.csv` and adds 7 broader medication bins corresponding to ETC codes of visible medications prescribed in the past year.  
- `labs.py` - Inputs `final_ecgs.csv` and adds 36 new features corresponding to 9 lab results visible before prediction cutoff. Each lab result contains columns for first value, peak value, abnormal, and missing.  
- `vitals.py` - Inputs `final_ecgs.csv` and creates a new csv labeled `vitals_long_cleaned.csv`. This csv contains rows for all vitals charted within the prediction cutoff corresponding to each stay in `final_ecgs.csv`. 'Pain' is simply imputed with '-1' for missing values and 2 flags for unable/sleeping and 1 flag for missing.  

---

## Pretraining (inputs: MIMIC-IV raw csvs; output: `pretrained_model.pth`)

- `Preprocessing.py` - The main preprocessing scripit for pretraining. Takes in all raw MIMIC-IV raw files and outputs `pretraining_ecgs.csv` to be used for training  
- `Pretrain.py` - Script for training the ResNet. Takes in `pretraining_ecgs.csv` and outputs `pretrained_model.pth` (model weights).  
- `Pretrained_model.py` - Pretrained model skeleton for reference or import.  
- `pretrained_model.pth` - Saved pretrained ResNet-18 weights to be transferred to disposition task.  

---

## Models

- `All_features_ROCPRC_data.npz` - combined ROC and PRC data stored for models trained on all features (used to generate graphs)  

**GRU_ECG_only.py – End-to-end trainer for the ECG-only admission model.**  
• Inputs: `final_ecgs.csv`, raw waveform files in `..\ecg\`, and the pretrained ResNet weights `pretrained_model.pth`.    
  – Encodes up to six 12-lead ECGs per stay with the unfrozen ResNet-18 + linear adapter.  
  – Appends a time-delta channel, feeds the sequence to a single-layer GRU head, and optimizes BCE with class weighting.  
  – Trains two regimes (“All stays” ≥1 ECG; “MULTI stays” ≥2 ECGs), applies early stopping, and logs loss curves, AUROC/AUPRC, confusion matrices, and per-epoch metrics.  
• Outputs (in `outputs/gru/…`): best-epoch model checkpoints (`*_best.pth`), `*_roc_prc.npz` for plotting, loss-curve PNGs, and CSVs with epoch-level metrics & confusion matrices.  

**GRU_all_features.py – Trainer for the multimodal model that fuses ECGs, vitals, and scalar features.**  
• Inputs: `final_ecgs.csv`, `vitals_long_cleaned.csv`, waveform files, and `pretrained_model.pth`.  
  – Generates ECG embeddings exactly as above, **plus** embeds up to ten rows of sequential vitals and 353 z-scored / boolean scalar features.  
  – Summarizes ECG and vital sequences with separate GRUs, concatenates both hidden states with the scalar vector, and passes through a dropout-FC head.  
  – Trains “All stays” and “MULTI stays” variants, handles imputation and z-scaling, performs class-weighted BCE training with early stopping, and records full performance diagnostics.  
• Outputs (in `all_performance/`, `multi_performance/`, and `combined_performance/`): best-epoch `*_best_model.pth`, `*_roc_prc_data.npz`, loss and ROC/PRC figures, epoch-metric CSVs, and confusion-matrix summaries.  

- `all_stays_ECGonly.pth` - Saved model weights for ECG-only features trained on full cohort (AUROC 0.845 AUPRC 0.863)  
- `all_stays_ECGonly_ROCPRC.npz` - Corresponding ROC and PRC data for `all_stays_ECGonly.pth`  

- `all_stays_model.pth` - Saved model weights for all features trained on full cohort (AUROC 0.913 AUPRC 0.932)  

- `multi_stays_ECGonly.pth` - Saved model weights for ECG-only features trained on subset ≥2 ECG cohort (AUROC 0.878 AUPRC 0.918)  
- `all_stays_ECGonly_ROCPRC.npz` - Corresponding ROC and PRC data for `multi_stays_ECGonly.pth`  

- `multiple_ECG_stays_model.pth` - Saved model weights for all features trained on subset ≥2 ECG cohort (AUROC 0.934 AUPRC 0.960)  
