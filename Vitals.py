import pandas as pd

# ---------- paths ----------
ecg_csv     = "sequential_ecgs.csv"
triage_csv  = "triage.csv.gz"
vitals_csv  = "vitalsign.csv.gz"
out_csv     = "vitals_long.csv"

# ---------- load once ----------
ecg_df     = pd.read_csv(ecg_csv,    parse_dates=['ecg_time'])
triage_df  = pd.read_csv(triage_csv)
vitals_df  = pd.read_csv(vitals_csv, parse_dates=['charttime'])

print(f"Loaded:")
print(f"  ECG rows      : {len(ecg_df):,}  (unique stays: {ecg_df['stay_id'].nunique():,})")
print(f"  Triage rows   : {len(triage_df):,}  (unique stays: {triage_df['stay_id'].nunique():,})")
print(f"  Vitals rows   : {len(vitals_df):,}  (unique stays: {vitals_df['stay_id'].nunique():,})\n")

# ---------- keep only stay_ids that appear in the ECG file ----------
keep_ids = set(ecg_df['stay_id'])
triage_df  = triage_df [triage_df ['stay_id'].isin(keep_ids)].copy()
vitals_df  = vitals_df[vitals_df['stay_id'].isin(keep_ids)].copy()

print(f"After aligning to ECG stays:")
print(f"  Triage rows   : {len(triage_df):,}  (unique stays: {triage_df['stay_id'].nunique():,})")
print(f"  Vitals rows   : {len(vitals_df):,}  (unique stays: {vitals_df['stay_id'].nunique():,})\n")

# ---------- get last ECG time per stay ----------
last_ecg = (
    ecg_df
    .groupby('stay_id', as_index=False)['ecg_time']
    .max()
    .rename(columns={'ecg_time': 'last_ecg_time'})
)

# Trim vitals *after* last ECG
pre_trim_stays = vitals_df['stay_id'].nunique()
vitals_df = vitals_df.merge(last_ecg, on='stay_id', how='left')
vitals_df = vitals_df[vitals_df['charttime'] <= vitals_df['last_ecg_time']]
vitals_df = vitals_df.drop(columns='last_ecg_time')

print(f"After dropping vitals recorded AFTER last-ECG:")
print(f"  Vitals rows   : {len(vitals_df):,}  (unique stays: {vitals_df['stay_id'].nunique():,})")
dropped = pre_trim_stays - vitals_df['stay_id'].nunique()
print(f"  Stays removed : {dropped:,}\n")

# ---------- unify column order ----------
cols = ["stay_id", "source", "charttime",
        "temperature", "heartrate", "resprate",
        "o2sat", "sbp", "dbp", "pain"]

# (1) TRIAGE rows
triage_long = (
    triage_df.assign(source="triage", charttime=pd.NaT)[cols]
)

# (2) PERIODIC rows
vitals_long = vitals_df.assign(source="periodic")[cols]

# ---------- concatenate & save ----------
long_df = (
    pd.concat([triage_long, vitals_long], ignore_index=True)
      .sort_values(["stay_id", "charttime", "source"])
)

print(f"Final long file: {len(long_df):,} rows, "
      f"{long_df['stay_id'].nunique():,} unique stays.")
# ── convert charttime to proper datetime (if not done already) ────────────────
long_df["charttime"] = pd.to_datetime(long_df["charttime"])

# ── sort each stay chronologically & give every row an order index ────────────
long_df = long_df.sort_values(["stay_id", "charttime"])

# replace the old “source” column with 1-based order: 1 = triage, 2, 3, …
long_df["source"] = long_df.groupby("stay_id").cumcount() + 1
# Treat empty strings or real NaNs as missing, replace with 0
long_df.loc[long_df["charttime"].isin(["", None]) | long_df["charttime"].isna(),
            "source"] = 0

long_df = long_df.sort_values(["stay_id", "source"])
long_df['source'] = long_df['source'] + 1
long_df.to_csv(out_csv, index=False)
print(f"\n✓ Wrote {out_csv}")
import pandas as pd
df = pd.read_csv("vitals_long.csv")
# assume df is your DataFrame
col = "pain"
import re

df[col] = df[col].astype(str).str.strip().str.lower()

# ---------------------------------------------------------------------------
# 1.  flag “sleep” and “unable/uta/ua”  ➜   indicator columns + set pain = -1
# ---------------------------------------------------------------------------
sleep_mask = df[col].str.contains(r"sleep",             na=False)
uta_mask   = df[col].str.contains(r"\b(?:unable|uta|ua)\b", na=False)

df["sleeping"] = sleep_mask.astype(int)
df["uta"]      = uta_mask.astype(int)

df.loc[sleep_mask | uta_mask, col] = -1                 # force value –1


  # 4-6, 2 - 5 …

slash_pat = re.compile(r"\s*(\d*\.?\d+|\.\d+)\s*/\s*10\s*$")        # 3/10, .4/10 …
dash_pat  = re.compile(r"\s*(\d+)\s*-\s*(\d+)\s*$")               

df["pain"] = df["pain"].apply(
    lambda v: (
        (lambda m: (int(m.group(1)) + int(m.group(2))) / 2)(dash_pat.fullmatch(str(v)))
        if dash_pat.fullmatch(str(v))
        else (lambda m: float(m.group(1)))(slash_pat.fullmatch(str(v)))
        if slash_pat.fullmatch(str(v))
        else v                         # leave unchanged if neither pattern matches
    )
)

leftover_mask = pd.to_numeric(df[col], errors="coerce").isna()   # still text or blank
df.loc[leftover_mask, col] = -1

df["real_pain"] = (df[col] != -1).astype(int)
is_non_numeric_str = (
    pd.to_numeric(df[col], errors="coerce").isna()   # couldn’t convert to number
    & df[col].notna()                                # and isn’t actual NaN
)

# 2️⃣ Get counts
vc = df.loc[is_non_numeric_str, col].value_counts(dropna=False)


# 3️⃣ Pretty-print
for val, n in vc.items():
    print(f"{repr(val):<30} → {n} rows")


display(df.head(50))
print(len(df))
df.to_csv('vitals_long.csv', index=False)

#!/usr/bin/env python
# clean_vitals.py
import pandas as pd

CSV_IN  = "vitals_long.csv"
CSV_OUT = "vitals_long_cleaned.csv"

VITAL_COLS = [
    "temperature", "heartrate", "resprate",
    "o2sat", "sbp", "dbp", "pain"
]

print(f"[INFO] reading {CSV_IN} …")
df = pd.read_csv(CSV_IN)

# make sure 'source' is an integer column
df["source"] = pd.to_numeric(df["source"], errors="raise").astype("Int64")

def clean_group(g: pd.DataFrame) -> pd.DataFrame:
    """If source-1 and source-2 rows are value-wise identical, drop the first."""
    g = g.sort_values("source").copy()

    if {1, 2}.issubset(g["source"].values):
        r1 = g.loc[g["source"] == 1, VITAL_COLS].iloc[0]
        r2 = g.loc[g["source"] == 2, VITAL_COLS].iloc[0]

        if r1.equals(r2):
            # drop the source-1 row
            g = g[g["source"] != 1].copy()
            # shift all remaining sources down by one
            g.loc[g["source"] > 1, "source"] -= 1
    return g

print("[INFO] scanning for duplicate (1,2) rows …")
cleaned = (
    df.groupby("stay_id", group_keys=False)
      .apply(clean_group)
      .reset_index(drop=True)
)

dropped = len(df) - len(cleaned)
print(f"[DONE] removed {dropped} duplicate rows "
      f"({dropped / len(df):.2%} of the file)")

print(f"[INFO] writing cleaned file → {CSV_OUT}")
cleaned.to_csv(CSV_OUT, index=False)


