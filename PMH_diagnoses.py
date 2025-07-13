import pandas as pd

# ───────────────────────────────────────────────────────────────
# 1.  LOAD stay table and source files
# ───────────────────────────────────────────────────────────────
df   = pd.read_csv("final_ecgs.csv", parse_dates=["intime"])              # stay_id, subject_id, intime
adm  = pd.read_csv("admissions.csv.gz",  parse_dates=["dischtime"])
diag = pd.read_csv("diagnoses_icd.csv.gz",   dtype={"icd_code":"string"})

# ICD-9 single-level CCS cross-walk  (numeric 001-281)
# ICD-9 single-level CCS – take just the first 2 columns, ignore headers
ccs9 = pd.read_csv(
    "$DXREF 2008_Archive.csv",   # literal filename with the $
    sep=",",                     # the file is comma-delimited
    quotechar="'",               # values wrapped in single quotes
    skiprows=1,                  # skip the “NOTE: New codes …” line
    header=None,                 # treat first data row as row 0
    usecols=[0, 1],              # keep only the first 2 columns
    dtype=str
)

# now rename and tidy
ccs9.columns = ["icd_code", "bin"]          # give explicit names
ccs9["icd_code"] = ccs9["icd_code"].str.strip().str.zfill(5)
ccs9["bin"]      = ccs9["bin"].str.strip().str.zfill(3)

# ICD-10 CCSR reference  (alphanumeric e.g. INJ041 → INJ group)
ccsr = (
    pd.read_excel(
        "DXCCSR-Reference-File-v2025-1.xlsx",            # the standard sheet in every release
        dtype=str,
        sheet_name="DX_to_CCSR_Mapping",
        usecols=[0,2]    # CCSR1 is the primary category
    )
      
)
ccsr.columns = ["icd_code", "bin"]          # give explicit names
ccsr["icd_code"] = ccsr["icd_code"].str.strip().str.zfill(5)
ccsr["bin"]      = ccsr["bin"].str.strip().str.zfill(3)
# ───────────────────────────────────────────────────────────────
# 2.  NORMALISE ICD-10 bins to 3-letter groups  (CIR, INJ, …)
# ───────────────────────────────────────────────────────────────
ccsr["bin"] = ccsr["bin"].str[:3]          # keep only body-system prefix

# ───────────────────────────────────────────────────────────────
# 3.  Concatenate both maps into one lookup  (icd_code → bin)
# ───────────────────────────────────────────────────────────────
lkp = pd.concat([ccs9, ccsr]).drop_duplicates("icd_code")

# ───────────────────────────────────────────────────────────────
# 4.  LIMIT diagnoses to admissions discharged *before* this stay’s intime
# ───────────────────────────────────────────────────────────────
diag = diag.merge(df[["subject_id"]].drop_duplicates(), on="subject_id")

adm  = adm.merge(df[["subject_id", "intime"]], on="subject_id")
adm  = adm[adm["dischtime"] < adm["intime"]]

diag = diag[diag["hadm_id"].isin(adm["hadm_id"])]

# ───────────────────────────────────────────────────────────────
# 5.  Map codes → bin  and pivot one-hot flags
# ───────────────────────────────────────────────────────────────
diag = diag.merge(lkp, on="icd_code", how="left").dropna(subset=["bin"])
diag["flag"] = 1

pmh = (diag.pivot_table(index="subject_id",
                        columns="bin",
                        values="flag",
                        aggfunc="max",
                        fill_value=0)
          .add_prefix("pmh_")              # e.g. pmh_CIR , pmh_041
          .reset_index())

pmh["n_pmh_bins"] = diag.groupby("subject_id").size().values

# ───────────────────────────────────────────────────────────────
# 6.  MERGE into final_ecgs  (subject_id unique in df)
# ───────────────────────────────────────────────────────────────
df = (df.merge(pmh, on="subject_id", how="left")
        .fillna(0))

print("Added PMH columns:", len([c for c in df.columns if c.startswith("pmh_")]))
print("Stays with ≥1 PMH bin:", (df["n_pmh_bins"]>0).mean()*100, "%")
