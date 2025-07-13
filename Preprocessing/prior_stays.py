''' Adds prior stay information (disposition and number of stays) to the final csv'''

final_ecgs = pd.read_csv("final_ecgs.csv",           
                         parse_dates=["ecg_time"])

edstays = pd.read_csv("edstays.csv.gz",         
                      parse_dates=["intime", "outtime"])

fix_lookup = edstays.set_index("stay_id")["intime"]



# 2) map that lookup onto df and keep the original when no correction exists
final_ecgs["intime"] = final_ecgs["stay_id"].map(fix_lookup).combine_first(final_ecgs["intime"])
# Ensure a subject_id column exists in both
assert {"subject_id", "stay_id", "disposition", "outtime"}.issubset(edstays.columns)
assert "subject_id" in final_ecgs.columns

# ------------------------------------------------------------------
# 1.  For speed: sort ED stays by subject and outtime
# ------------------------------------------------------------------
edstays_sorted = edstays.sort_values(["subject_id", "outtime"])

# ------------------------------------------------------------------
# 2.  Define a helper that finds the previous stay for ONE row
# ------------------------------------------------------------------
def previous_stay(row):
    subj_stays = edstays_sorted[edstays_sorted["subject_id"] == row["subject_id"]]
    prev = subj_stays[subj_stays["outtime"] < row["ecg_time"]].tail(1)  # closest before

    if prev.empty:
        return "first"                       # no prior stay
    elif prev["disposition"].iloc[0] == "HOME":
        return "home"
    else:
        return "else"

# ------------------------------------------------------------------
# 3.  Apply row-wise and expand into three Boolean columns
# ------------------------------------------------------------------
prev_status = final_ecgs.apply(previous_stay, axis=1)

final_ecgs["previous_home"]  = (prev_status == "home").astype(int)
final_ecgs["previous_else"]  = (prev_status == "else").astype(int)
final_ecgs["first_stay"]     = (prev_status == "first").astype(int)

# ------------------------------------------------------------------
# 4.  (Optional) sanity check & save
# ------------------------------------------------------------------
print(final_ecgs[["stay_id", "previous_home", "previous_else", "first_stay"]].head())
total = len(final_ecgs)

for col in ["previous_home", "previous_else", "first_stay"]:
    pct = 100 * final_ecgs[col].mean()          # mean of 0/1 column = prevalence
    print(f"{col:<15}: {pct:5.2f}% of stays")

# -------------------------------------------------------------
# 2.  Sanity check: no row should have >1 flag set
# -------------------------------------------------------------
max_flags = final_ecgs[["previous_home",
                        "previous_else",
                        "first_stay"]].sum(axis=1).max()

assert max_flags <= 1, "⚠️  Some rows have more than one flag set!"
print("\nSanity check passed – each stay has exactly one flag.")
import pandas as pd

# ── 1 · load tables ────────────────────────────────────────────
final_ecgs = pd.read_csv("final_ecgs.csv", parse_dates=["intime"])
edstays    = pd.read_csv("edstays.csv.gz",
                         usecols=["subject_id", "stay_id", "outtime"],
                         parse_dates=["outtime"])

# ── 2 · merge & filter to stays discharged BEFORE this intime ──
merged = (
    final_ecgs[["stay_id", "subject_id", "intime"]]
      .merge(edstays,
             on="subject_id",
             how="left",
             suffixes=("", "_prior"))       # current vs. prior stay cols
)

mask = merged["outtime"] < merged["intime"]           # prior stays only
merged_prior = merged.loc[mask]

# ── 3 · count prior stays and align back ───────────────────────
prior_counts = (
    merged_prior.groupby("stay_id")["stay_id_prior"]   # any column works
               .size()
               .reindex(final_ecgs["stay_id"], fill_value=0)
               .astype(int)
)

final_ecgs["n_prior_stays"] = prior_counts.values

# quick sanity
display(final_ecgs.head(30))

final_ecgs.to_csv('final_ecgs.csv', index = False)
