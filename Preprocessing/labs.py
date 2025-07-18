"""
LAB-FEATURE PIPELINE  (multi-itemid groups + abnormal flag)
────────────────────────────────────────────────────────────
Adds **four columns per lab group** to final_ecgs:

    <label>_first     = earliest numeric result between intime and ecg_time
    <label>_peak      = max numeric result in that window
    <label>_abn       = 1 ⇢ any row’s labevents.flag == "abnormal"
    <label>_missing   = 1 ⇢ no row at all for that label (else 0)
    
"""

import pandas as pd
import numpy as np
                # folder containing the two input files

LAB_GROUPS = {                       # label : [itemid, itemid, …]
    "tropT": [51003],                           # Troponin-T
    "lact" : [53154, 50813, 52442],             # Lactate (venous + wb)
    "hgb"  : [51645, 51640, 50811],             # Hemoglobin variants
    "creat": [50912],                           # Creatinine
    "bnp"  : [50963],                           # BNP / NT-proBNP
    "k"    : [50822, 50971],                    # Potassium
    "mg"   : [50960, 51088],                    # Magnesium
    "wbc"  : [51300, 51516],                    # White-cell count
    "crp"  : [50889],                           # C-reactive protein
}

# ───────────────────────────────────────────────────────────────
# 1 · LOAD TABLES
# ───────────────────────────────────────────────────────────────
final_ecgs = pd.read_csv(
    "final_ecgs.csv",
    parse_dates=["intime", "ecg_time"]
)

labs = pd.read_csv(
   "labevents.csv.gz",
    usecols=["subject_id", "itemid", "charttime",
             "value", "valuenum", "valueuom", "flag"],
    dtype={"subject_id": "int32", "itemid": "int32"},
    parse_dates=["charttime"]
)

# ───────────────────────────────────────────────────────────────
# 2 · FILTER TO NEEDED ITEMIDS & TIME-GATE
# ───────────────────────────────────────────────────────────────
all_ids = [iid for ids in LAB_GROUPS.values() for iid in ids]
labs = labs[labs["itemid"].isin(all_ids)]

labs = labs.merge(
    final_ecgs[["subject_id", "intime", "ecg_time"]],
    on="subject_id", how="inner"
)

labs = labs[(labs["charttime"] >= labs["intime"]) &
            (labs["charttime"] <= labs["ecg_time"])]

# ───────────────────────────────────────────────────────────────
# 3 · COERCE NUMERIC RESULT  (handles swapped columns)
# ───────────────────────────────────────────────────────────────
def to_float(row):
    if pd.notna(row["valuenum"]):
        return row["valuenum"]
    try:
        return float(row["value"])
    except (TypeError, ValueError):
        return 0.0                      # below detection

labs["num"] = labs.apply(to_float, axis=1)
labs["flag_clean"] = labs["flag"].astype(str).str.strip().str.lower()

# ───────────────────────────────────────────────────────────────
# 4 · BUILD FEATURES PER GROUP
# ───────────────────────────────────────────────────────────────
frames = []

for label, ids in LAB_GROUPS.items():
    grp = labs[labs["itemid"].isin(ids)]

    first = (grp.sort_values("charttime")
                  .groupby("subject_id")["num"].first())
    peak  = grp.groupby("subject_id")["num"].max()
    abn   = (grp.groupby("subject_id")["flag_clean"]
                 .apply(lambda s: int((s == "abnormal").any())))

    feat = pd.DataFrame({
        f"{label}_first"   : first,
        f"{label}_peak"    : peak,
        f"{label}_abn"     : abn
    })

    feat[f"{label}_missing"] = feat[f"{label}_first"].isna().astype(int)
    feat[[f"{label}_first", f"{label}_peak"]] = (
        feat[[f"{label}_first", f"{label}_peak"]].fillna(0)
    )

    frames.append(feat)

lab_features = (
    pd.concat(frames, axis=1)
      .reset_index()
      .rename(columns={"index": "subject_id"})
)

# ───────────────────────────────────────────────────────────────
# 5 · MERGE & FINAL NaN HANDLING
# ───────────────────────────────────────────────────────────────
final_ecgs = final_ecgs.merge(lab_features, on="subject_id", how="left")

for lab in LAB_GROUPS:
    final_ecgs[f"{lab}_missing"] = final_ecgs[f"{lab}_missing"].fillna(1)
    for col in (f"{lab}_first", f"{lab}_peak", f"{lab}_abn"):
        final_ecgs[col] = final_ecgs[col].fillna(0)

# ───────────────────────────────────────────────────────────────
# 6 · SAVE
# ───────────────────────────────────────────────────────────────
out_path =  "final_ecgs.csv"
final_ecgs.to_csv(out_path, index=False)

print(f"✅  Added {len(LAB_GROUPS)*4} lab-feature columns and wrote {out_path.name}.")
