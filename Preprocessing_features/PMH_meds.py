"""
OPTION A  ─  Chronic‐medication flags from prescriptions.csv
------------------------------------------------------------

Window   : 365 days BEFORE each ED stay’s `intime`
Features : one-hot flags (ETC-level-2, prefixed hxETC2_)  +  n_chronic_meds scalar
Leakage  : none  (all rows strictly before ED arrival)
"""

import pandas as pd

# ───────────────────────────────────────────────────────────────
# 1.  LOAD STAY TABLE  (needs stay_id, subject_id, intime)
# ───────────────────────────────────────────────────────────────
final_ecgs = pd.read_csv("final_ecgs.csv", parse_dates=["intime"])

# ───────────────────────────────────────────────────────────────
# 2.  LOAD SOURCE TABLES
# ───────────────────────────────────────────────────────────────
rx = pd.read_csv(
    "prescriptions.csv.gz",
    usecols=["subject_id", "starttime", "stoptime", "gsn"],
    dtype={"gsn": "string"},
    parse_dates=["starttime", "stoptime"],
)

medrec = pd.read_csv(
    "medrecon.csv.gz",
    usecols=["gsn", "etccode"],
    dtype={"gsn": "string", "etccode": "string"},
)

# ───────────────────────────────────────────────────────────────
# 3.  ZERO-PAD GSNs  (always 6 digits)
# ───────────────────────────────────────────────────────────────
rx["gsn"]      = rx["gsn"].str.zfill(6)
medrec["gsn"]  = medrec["gsn"].str.zfill(6)

# ───────────────────────────────────────────────────────────────
# 4.  BUILD  gsn → etc  LOOKUP   (one row per gsn)
# ───────────────────────────────────────────────────────────────
gsn2etc = (
    medrec.dropna(subset=["gsn", "etccode"])
          .drop_duplicates("gsn")
          .set_index("gsn")["etccode"]
)

# ───────────────────────────────────────────────────────────────
# 5.  FILTER TO 1-YEAR HISTORY  (strictly before ED arrival)
# ───────────────────────────────────────────────────────────────
hist = rx.merge(final_ecgs[["stay_id", "subject_id", "intime"]],
                on="subject_id", how="inner")

mask = (
    (hist["starttime"] < hist["intime"]) &
    (hist["starttime"] >= hist["intime"] - pd.Timedelta(days=365))
)
hist = hist.loc[mask].copy()

# ───────────────────────────────────────────────────────────────
# 6.  MAP TO ETC-L2  &  PIVOT TO COUNTS
# ───────────────────────────────────────────────────────────────
hist["etcL4"] = hist["gsn"].map(gsn2etc)
hist = hist.dropna(subset=["etcL4"])

hist["etcL2"] = hist["etcL4"].str.slice(0, 5) + "000"   # 5 digits + "000"
hist["flag"]  = 1

hx_flags = (
    hist.pivot_table(index="stay_id",
                     columns="etcL2",
                     values="flag",
                     aggfunc="sum",
                     fill_value=0)
        .add_prefix("hxETC2_")           # e.g. hxETC2_00007000
        .reset_index()
)

hx_flags["n_chronic_meds"] = hist.groupby("stay_id").size().values

# ───────────────────────────────────────────────────────────────
# 7.  MERGE INTO MASTER  (Pyxis flags remain untouched)
# ───────────────────────────────────────────────────────────────
final_ecgs = (
    final_ecgs.merge(hx_flags, on="stay_id", how="left")
              .fillna(0)
)

# ───────────────────────────────────────────────────────────────
# 8.  SANITY PRINT
# ───────────────────────────────────────────────────────────────
added_cols = hx_flags.shape[1] - 2   # minus stay_id & n_chronic_meds
coverage   = (final_ecgs["n_chronic_meds"] > 0).mean() * 100

print(f"Chronic-med class columns added : {added_cols}")
print(f"Stays with ≥1 chronic med      : {coverage:.1f}%")

final_ecgs.to_csv('final_ecgs.csv', index=False)
