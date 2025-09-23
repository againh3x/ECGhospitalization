# Serial 12-Lead ECG–Based Deep-Learning Model for Real-Time Prediction of Hospital Admission in Emergency-Department Cardiac Presentations: Retrospective Cohort Study  
**Arda Altintepe**

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

Three modelling approaches are provided in the `Models/` directory. All are trained using **stratified 5-fold cross-validation** on two cohorts:  

- **All stays** (≥1 ECG)  
- **Multi-ECG subset** (≥2 ECGs per encounter)

---

### 1. ECG-only GRU  
`Models/ECG only/GRU_ecg.py`  

- End-to-end trainer that encodes up to six 12-lead waveforms per stay with an **unfrozen ResNet-18 encoder + linear adapter**, appends time-delta information, and processes the sequence with a **GRU**.  
- **Performance:** AUROC ≈ **0.852** (all stays), **0.859** (≥2-ECG subset).  
- **Outputs:** per-fold metrics, pooled ROC curves, CSV summaries in:  
  - `Models/ECG only/Metrics - All Stays`  
  - `Models/ECG only/Metrics - Multi-ECG Stays`

---

### 2. Tabular baseline  
`Models/Tabular/train.py`  

- Random-forest classifier trained on **demographic, triage, medication, past medical history, and labs** available up to prediction time.  
- **Performance:** AUROC ≈ **0.886** (all stays), **0.911** (≥2-ECG subset).  
- **Outputs:** per-fold metrics, cross-validation summaries in:  
  - `Models/Tabular/Metrics - All Stays`  
  - `Models/Tabular/Metrics - Multi-ECG Stays`

---

### 3. Multimodal fusion  
`Models/Multimodal/train.py`  

- Fuses **ECG embeddings, sequential vital signs, and static features** from the tabular model.  
- ECG and vital sequences are separately summarised with **recurrent networks**, concatenated with static features, then passed through a **dropout-FC head**.  
- **Performance:** AUROC ≈ **0.911** (all stays), **0.924** (≥2-ECG subset).  
- **Outputs:** per-fold metrics, loss and ROC figures, pooled ROC/PR curves, and summaries in:  
  - `Models/Multimodal/Metrics - All Stays`  
  - `Models/Multimodal/Metrics - Multi-ECG Stays`

---

### Cross-validation summaries

- Files named `*_cv_summary.csv` report **mean ± std AUROC across folds** and the **pooled AUROC**.  
- Out-of-fold predictions (`*_val_predictions.csv`) are provided for each fold to support **plotting and statistical comparisons**.  
- All scripts reuse the **same stratified splits** to enable fair model comparison.
