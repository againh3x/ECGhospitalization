import pandas as pd

# ────────────────────────────────────────────────────────────────
# 1.  LOAD TABLES
# ────────────────────────────────────────────────────────────────
final_ecgs = pd.read_csv(
    "final_ecgs.csv",
    parse_dates=["intime", "ecg_time"]
)

pyxis = pd.read_csv(
    "pyxis.csv.gz",
    usecols=["stay_id", "charttime", "gsn"],
    dtype={"gsn": "string"},
    parse_dates=["charttime"]
)

medrec = pd.read_csv(
    "medrecon.csv.gz",
    usecols=["gsn", "etccode", "etcdescription"],
    dtype={"gsn": "string", "etccode": "string"}
)

# ────────────────────────────────────────────────────────────────
# 2.  GSN → ETC LOOKUP (from medrecon)
# ────────────────────────────────────────────────────────────────
gsn2etc = (medrec.dropna(subset=["gsn", "etccode"])
                  .drop_duplicates(subset=["gsn"])
                  .set_index("gsn")[["etccode"]])

# ────────────────────────────────────────────────────────────────
# 3.  MERGE ETC INTO PYXIS & TIME-GATE (≤ last ECG)
# ────────────────────────────────────────────────────────────────
pyxis = (pyxis.merge(gsn2etc, left_on="gsn", right_index=True, how="left")
               .dropna(subset=["etccode"])
               .merge(final_ecgs[["stay_id", "ecg_time"]], on="stay_id", how="inner")
               .query("charttime <= ecg_time")
               .copy())

# ────────────────────────────────────────────────────────────────
# 4.  CREATE HIERARCHY LEVELS  (L4, L3, L2)
# ────────────────────────────────────────────────────────────────

pyxis["etcL4"] = pyxis["etccode"].str.zfill(8)        # 8-digit as-is
pyxis["etcL3"] = pyxis["etcL4"].str.slice(0, 6) + "00"   # 6 + "00"  → 8-d
pyxis["etcL2"] = pyxis["etcL4"].str.slice(0, 5) + "000"  # 5 + "000" → 8-d

def pivot_flags(df: pd.DataFrame, col: str, prefix: str) -> pd.DataFrame:
    return (df.assign(flag=1)
              .pivot_table(index="stay_id",
                           columns=col,
                           values="flag",
                           aggfunc="max",
                           fill_value=0)
              .add_prefix(prefix))

flags_L3 = pivot_flags(pyxis, "etcL3", "etc3_")     # ~120 cols
flags_L2 = pivot_flags(pyxis, "etcL2", "etc2_")     # ~40–60 cols

flags = flags_L3.join(flags_L2).reset_index()

# total Pyxis dispenses in causal window
flags = flags.merge(
            pyxis.groupby("stay_id").size()
                 .rename("n_pyxis_total")
                 .to_frame()
                 .reset_index(),
            on="stay_id", how="left"
        )

# ────────────────────────────────────────────────────────────────
# 5.  MERGE INTO MASTER & FILL ZEROS
# ────────────────────────────────────────────────────────────────
final_ecgs = (final_ecgs
              .merge(flags, on="stay_id", how="left")
              .fillna(0))

# ────────────────────────────────────────────────────────────────
# 6.  SHOW ALL COLUMNS (no truncation) & PREVALENCE SUMMARY
# ────────────────────────────────────────────────────────────────
pd.set_option("display.max_columns", None)

flag_cols = [c for c in final_ecgs.columns if c.startswith(("etc3_", "etc2_"))]
top10 = (final_ecgs[flag_cols].sum()
                     .sort_values(ascending=False)
                     .head(10) / len(final_ecgs) * 100)

print("Top 10 ETC flags (≤ last ECG):")
print((top10.round(2).astype(str) + " %").to_string())

print(len(final_ecgs))
final_ecgs.to_csv('final_ecgs.csv', index = False)
