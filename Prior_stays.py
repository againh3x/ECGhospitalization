final_ecgs = pd.read_csv("final_ecgs.csv",               # or use the DF already in memory
                         parse_dates=["ecg_time"])

edstays = pd.read_csv("edstays.csv.gz",                     # the file you’ll upload
                      parse_dates=["intime", "outtime"]) # cols typically present

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

final_ecgs.to_csv('final_ecgs.csv', index = False)
