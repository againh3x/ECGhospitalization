
"""
RandomForest (tabular-only) baseline — 5-fold CV ONLY (no held-out test)
Reuses the identical StratifiedKFold splits as the multimodal script:
  • "MULTI stays" (n_ecg >= 2)
  • "All stays"   (all stays)
"""

import os, json, warnings, random
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    roc_curve, precision_recall_curve,
    confusion_matrix, precision_recall_fscore_support, accuracy_score
)

# RandomForest + joblib (for saving/loading)
from sklearn.ensemble import RandomForestClassifier
from joblib import dump

# Matplotlib optional
try:
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except Exception:
    HAVE_MPL = False

warnings.filterwarnings("ignore", category=UserWarning)

# ────────────────────────────────────────────────────────────────────────────
# Files / columns
# ────────────────────────────────────────────────────────────────────────────
CSV_ECG = "final_ecgs.csv"

NUM_SCALARS = [
    "n_ecg", "pre_los", "acuity", "age", "n_pyxis_total",
    "hxETC2_00000000", "hxETC2_00001000", "hxETC2_00002000", "hxETC2_00003000",
    "hxETC2_00004000", "hxETC2_00005000", "hxETC2_00006000", "n_chronic_meds",
    "n_pmh_bins", "n_prior_stays", "tropT_first", "tropT_peak", "lact_first",
    "lact_peak", "hgb_first", "hgb_peak", "creat_first", "creat_peak",
    "bnp_first", "bnp_peak", "k_first", "k_peak", "mg_first", "mg_peak",
    "wbc_first", "wbc_peak", "crp_first", "crp_peak"
]

BOOL_SCALARS = [
    "gender_M", "arrival_transport_AMBULANCE", "arrival_transport_HELICOPTER",
    "arrival_transport_WALK IN", "Chest Pain", "Dyspnea", "Presyncope", "Syncope",
    "previous_home", "previous_else", "first_stay", "etc3_00000000",
    "etc3_00000100", "etc3_00000200", "etc3_00000300", "etc3_00000400",
    "etc3_00000500", "etc3_00000600", "etc3_00000700", "etc3_00000800",
    "etc3_00000900", "etc3_00001000", "etc3_00001200", "etc3_00002500",
    "etc3_00002600", "etc3_00002700", "etc3_00002800", "etc3_00003000",
    "etc3_00003100", "etc3_00003600", "etc3_00003700", "etc3_00003900",
    "etc3_00004500", "etc3_00004600", "etc3_00005600", "etc3_00005700",
    "etc3_00005800", "etc3_00005900", "etc3_00006000", "etc3_00006100",
    "etc3_00006200", "etc3_00006400", "etc2_00000000", "etc2_00001000",
    "etc2_00002000", "etc2_00003000", "etc2_00004000", "etc2_00005000",
    "etc2_00006000",
    # (PMH + lab flags — same long list as in your multimodal script)
    "pmh_001", "pmh_002", "pmh_003", "pmh_004", "pmh_005", "pmh_006",
    "pmh_007", "pmh_008", "pmh_009", "pmh_010", "pmh_011", "pmh_012",
    "pmh_013", "pmh_014", "pmh_015", "pmh_016", "pmh_018", "pmh_019",
    "pmh_020", "pmh_022", "pmh_023", "pmh_025", "pmh_026", "pmh_027",
    "pmh_028", "pmh_029", "pmh_030", "pmh_031", "pmh_032", "pmh_033",
    "pmh_034", "pmh_035", "pmh_036", "pmh_037", "pmh_038", "pmh_039",
    "pmh_040", "pmh_041", "pmh_042", "pmh_044", "pmh_045", "pmh_047",
    "pmh_048", "pmh_049", "pmh_050", "pmh_051", "pmh_054", "pmh_055",
    "pmh_056", "pmh_057", "pmh_058", "pmh_059", "pmh_061", "pmh_062",
    "pmh_063", "pmh_064", "pmh_066", "pmh_067", "pmh_068", "pmh_069",
    "pmh_070", "pmh_072", "pmh_073", "pmh_074", "pmh_075", "pmh_076",
    "pmh_077", "pmh_078", "pmh_081", "pmh_082", "pmh_083", "pmh_084",
    "pmh_085", "pmh_086", "pmh_087", "pmh_088", "pmh_089", "pmh_090",
    "pmh_091", "pmh_092", "pmh_093", "pmh_094", "pmh_095", "pmh_096",
    "pmh_097", "pmh_099", "pmh_100", "pmh_101", "pmh_102", "pmh_103",
    "pmh_104", "pmh_105", "pmh_106", "pmh_107", "pmh_108", "pmh_109",
    "pmh_110", "pmh_113", "pmh_114", "pmh_115", "pmh_116", "pmh_117",
    "pmh_118", "pmh_121", "pmh_122", "pmh_124", "pmh_125", "pmh_126",
    "pmh_127", "pmh_128", "pmh_131", "pmh_133", "pmh_134", "pmh_136",
    "pmh_137", "pmh_138", "pmh_139", "pmh_140", "pmh_141", "pmh_143",
    "pmh_145", "pmh_146", "pmh_147", "pmh_148", "pmh_149", "pmh_151",
    "pmh_153", "pmh_155", "pmh_156", "pmh_159", "pmh_160", "pmh_161",
    "pmh_162", "pmh_163", "pmh_164", "pmh_165", "pmh_166", "pmh_167",
    "pmh_168", "pmh_170", "pmh_173", "pmh_175", "pmh_176", "pmh_177",
    "pmh_178", "pmh_180", "pmh_181", "pmh_182", "pmh_183", "pmh_184",
    "pmh_185", "pmh_186", "pmh_187", "pmh_188", "pmh_189", "pmh_190",
    "pmh_191", "pmh_192", "pmh_193", "pmh_195", "pmh_196", "pmh_197",
    "pmh_198", "pmh_199", "pmh_200", "pmh_201", "pmh_202", "pmh_203",
    "pmh_204", "pmh_205", "pmh_206", "pmh_207", "pmh_208", "pmh_209",
    "pmh_211", "pmh_212", "pmh_213", "pmh_214", "pmh_215", "pmh_216",
    "pmh_217", "pmh_224", "pmh_225", "pmh_226", "pmh_227", "pmh_228",
    "pmh_229", "pmh_230", "pmh_231", "pmh_232", "pmh_233", "pmh_234",
    "pmh_235", "pmh_236", "pmh_237", "pmh_238", "pmh_239", "pmh_240",
    "pmh_242", "pmh_243", "pmh_244",
    "pmh_248", "pmh_249", "pmh_250", "pmh_251", "pmh_252", "pmh_253",
    "pmh_254", "pmh_255", "pmh_256", "pmh_257", "pmh_258", "pmh_259",
    "pmh_2601", "pmh_2603", "pmh_2604", "pmh_2605", "pmh_2606", "pmh_2607",
    "pmh_2608", "pmh_2609", "pmh_2610", "pmh_2611", "pmh_2613", "pmh_2614",
    "pmh_2615", "pmh_2616", "pmh_2617", "pmh_2618", "pmh_2619", "pmh_2620",
    "pmh_2621", "pmh_BLD", "pmh_CIR", "pmh_DEN", "pmh_DIG", "pmh_EAR", "pmh_END",
    "pmh_EXT", "pmh_EYE", "pmh_FAC", "pmh_GEN", "pmh_INF", "pmh_INJ", "pmh_MAL",
    "pmh_MBD", "pmh_MUS", "pmh_NEO", "pmh_NVS", "pmh_PRG", "pmh_RSP", "pmh_SKN",
    "pmh_SYM",
    "tropT_abn", "tropT_missing", "lact_abn", "lact_missing", "hgb_abn",
    "hgb_missing", "creat_abn", "creat_missing", "bnp_abn", "bnp_missing",
    "k_abn", "k_missing", "mg_abn", "mg_missing", "wbc_abn", "wbc_missing",
    "crp_abn", "crp_missing",
]

ACUITY_MISS_FLAG = "missing_acuity"

RANDOM_STATE = 0
np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)

# ────────────────────────────────────────────────────────────────────────────
# Utilities
# ────────────────────────────────────────────────────────────────────────────

def roc_prc_arrays(y_true: np.ndarray, y_prob: np.ndarray):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = roc_auc_score(y_true, y_prob)
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)
    return fpr, tpr, roc_auc, prec, rec, pr_auc

def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5):
    fpr, tpr, roc_auc, prec, rec, pr_auc = roc_prc_arrays(y_true, y_prob)
    y_pred = (y_prob >= threshold).astype(int)
    acc = accuracy_score(y_true, y_pred)
    prc, rc, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    cm = confusion_matrix(y_true, y_pred).tolist()
    return {
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "accuracy": float(acc),
        "precision": float(prc),
        "recall": float(rc),
        "f1": float(f1),
        "cm": cm,
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "prec_curve": prec.tolist(),
        "rec_curve": rec.tolist(),
    }

def save_curves(metrics: dict, outpath_png_roc: Path, outpath_png_prc: Path):
    if not HAVE_MPL:
        return
    plt.figure()
    plt.plot(metrics["fpr"], metrics["tpr"], label=f"AUC {metrics['roc_auc']:.3f}")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("ROC curve (pooled)"); plt.legend(); plt.tight_layout()
    plt.savefig(outpath_png_roc, dpi=200); plt.close()

    plt.figure()
    plt.plot(metrics["rec_curve"], metrics["prec_curve"], label=f"AUPRC {metrics['pr_auc']:.3f}")
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title("Precision–Recall curve (pooled)"); plt.legend(); plt.tight_layout()
    plt.savefig(outpath_png_prc, dpi=200); plt.close()

# ────────────────────────────────────────────────────────────────────────────
# Data loading / features
# ────────────────────────────────────────────────────────────────────────────

def load_ecg_df(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["ecg_time", "intime"])
    if "disposition_HOME" not in df.columns:
        raise ValueError("final_ecgs.csv must contain 'disposition_HOME' (0/1).")
    # Match multimodal preprocessing for acuity
    df[ACUITY_MISS_FLAG] = df["acuity"].isna().astype(float)
    df["acuity"] = df["acuity"].fillna(df["acuity"].median())
    return df

def select_feature_columns(df: pd.DataFrame):
    feat_bool = BOOL_SCALARS + [ACUITY_MISS_FLAG]
    missing_num = [c for c in NUM_SCALARS if c not in df.columns]
    missing_bool = [c for c in feat_bool if c not in df.columns]
    if missing_num:
        print(f"[warn] missing numeric columns: {missing_num[:8]}{' ...' if len(missing_num)>8 else ''}")
    if missing_bool:
        print(f"[warn] missing boolean columns: {missing_bool[:8]}{' ...' if len(missing_bool)>8 else ''}")
    use_num = [c for c in NUM_SCALARS if c in df.columns]
    use_bool = [c for c in feat_bool if c in df.columns]
    return use_num + use_bool

# ────────────────────────────────────────────────────────────────────────────
# RandomForest helper (defaults only)
# ────────────────────────────────────────────────────────────────────────────

def make_rf() -> RandomForestClassifier:
    return RandomForestClassifier(
        random_state=RANDOM_STATE
        # keep sklearn defaults (n_estimators=100, max_depth=None, etc.)
        # NOTE: RF does not natively support NaN values.
    )

# ────────────────────────────────────────────────────────────────────────────
# Core: reproduce EXACT 5-fold splits from multimodal and train RF
# ────────────────────────────────────────────────────────────────────────────

def train_rf_variant(ecg_df: pd.DataFrame, keep_multi: bool, name: str):
    """
    Reproduce multimodal 5-fold StratifiedKFold and train RF baseline on tabular features only.
    NO held-out test set; report OOF (pooled) metrics and per-fold summaries.
    """
    stays = ecg_df.loc[ecg_df["n_ecg"].ge(2) if keep_multi else slice(None), "stay_id"].tolist()
    perf_dir = Path("rf_multi_performance" if keep_multi else "rf_all_performance")
    perf_dir.mkdir(exist_ok=True)

    y_all = ecg_df.set_index("stay_id").loc[stays, "disposition_HOME"].to_numpy().astype(int)

    # SAME 5-fold CV as multimodal on the FULL set
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    folds = list(skf.split(stays, y_all))

    feat_cols = select_feature_columns(ecg_df)
    X_table = ecg_df.set_index("stay_id")[feat_cols].copy().astype(float)
    y_table = ecg_df.set_index("stay_id")["disposition_HOME"].astype(int)

    # Fail fast if any NaNs remain (RF can't handle them)
    if np.isnan(X_table.values).any():
        n_nan = np.isnan(X_table.values).sum()
        raise ValueError(f"RF baseline: feature matrix contains {n_nan} NaNs. "
                         f"Please impute upstream (the multimodal workflow usually ensures this).")

    def get_Xy(ids):
        X = X_table.loc[ids]
        y = y_table.loc[ids].to_numpy().astype(int)
        return X, y

    all_val_preds = []
    fold_auc_list = []
    fold_ap_list  = []
    stem = name.lower().replace(" ", "_")

    # NEW: feature importance collector
    fold_importances = []  # list of np.array, one per fold  # NEW: feature importance

    for fold_idx, (tri, vai) in enumerate(folds):
        fold_tr_ids = [stays[i] for i in tri]
        fold_va_ids = [stays[i] for i in vai]
        X_tr, y_tr  = get_Xy(fold_tr_ids)
        X_va, y_va  = get_Xy(fold_va_ids)

        clf = make_rf()
        clf.fit(X_tr, y_tr)

        # NEW: collect feature importances for this fold
        if hasattr(clf, "feature_importances_"):  # NEW: feature importance
            fold_importances.append(clf.feature_importances_.astype(float))  # NEW: feature importance

        y_hat_va = clf.predict_proba(X_va)[:, 1]
        val_metrics = compute_metrics(y_va, y_hat_va)

        # Per-fold metrics CSV
        pd.DataFrame([{
            "fold": fold_idx,
            "roc_auc": val_metrics["roc_auc"],
            "pr_auc": val_metrics["pr_auc"],
            "accuracy": val_metrics["accuracy"],
            "precision": val_metrics["precision"],
            "recall": val_metrics["recall"],
            "f1": val_metrics["f1"],
        }]).to_csv(perf_dir / f"{stem}_fold{fold_idx}_metrics.csv", index=False)

        # Save RF per fold (optional)
        dump(clf, perf_dir / f"{stem}_fold{fold_idx}_rf.pkl")

        fold_auc_list.append(val_metrics["roc_auc"])
        fold_ap_list.append(val_metrics["pr_auc"])

        all_val_preds.append(pd.DataFrame({
            "stay_id": fold_va_ids,
            "true_label": y_va,
            "pred_prob": y_hat_va,
            "fold": fold_idx
        }))

    # OOF (pooled across folds)
    oof_df = pd.concat(all_val_preds, ignore_index=True)
    oof_df.to_csv(perf_dir / f"{stem}_val_predictions.csv", index=False)

    yt_all = oof_df["true_label"].to_numpy()
    yp_all = oof_df["pred_prob"].to_numpy()

    fpr, tpr, roc_auc, prec, rec, pr_auc = roc_prc_arrays(yt_all, yp_all)

    # CV summary
    summary = pd.DataFrame({
        "metric": ["val_auc_mean", "val_auc_std", "val_ap_mean", "val_ap_std", "pooled_auc", "pooled_ap", "n_samples"],
        "value":  [np.mean(fold_auc_list), np.std(fold_auc_list, ddof=1),
                   np.mean(fold_ap_list),  np.std(fold_ap_list,  ddof=1),
                   roc_auc, pr_auc, len(yt_all)]
    })
    summary.to_csv(perf_dir / f"{stem}_cv_summary.csv", index=False)

    # NEW: save aggregated feature importances across folds
    if len(fold_importances) > 0:  # NEW: feature importance
        fi_mat = np.vstack(fold_importances)                                 # NEW
        fi_mean = fi_mat.mean(axis=0)                                        # NEW
        fi_std  = fi_mat.std(axis=0, ddof=1) if fi_mat.shape[0] > 1 else np.zeros_like(fi_mean)  # NEW
        fi_df = pd.DataFrame({                                               # NEW
            "feature": feat_cols,
            "importance_mean": fi_mean,
            "importance_std": fi_std
        }).sort_values("importance_mean", ascending=False)                   # NEW
        fi_df.to_csv(perf_dir / f"{stem}_feature_importance.csv", index=False)  # NEW

    # Save pooled curves (per variant)
    np.savez(perf_dir / f"{stem}_cv_roc_prc_data.npz",
             fpr=fpr, tpr=tpr, auc=roc_auc, prec=prec, rec=rec, pr=pr_auc)

    if HAVE_MPL:
        plt.figure()
        plt.plot(fpr, tpr, label=f"{name} (pooled AUROC {roc_auc:.3f})")
        plt.plot([0,1],[0,1],'k--', alpha=.4)
        plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
        plt.title(f"{name} – RF CV ROC (pooled)"); plt.legend()
        plt.tight_layout(); plt.savefig(perf_dir / f"{stem}_cv_roc.png", dpi=200); plt.close()

        plt.figure()
        plt.plot(rec, prec, label=f"{name} (pooled AUPRC {pr_auc:.3f})")
        plt.xlabel("Recall"); plt.ylabel("Precision")
        plt.title(f"{name} – RF CV PR (pooled)"); plt.legend()
        plt.tight_layout(); plt.savefig(perf_dir / f"{stem}_cv_prc.png", dpi=200); plt.close()

    return {
        "name": name,
        "cv_mean_auc": float(np.mean(fold_auc_list)),
        "cv_mean_ap":  float(np.mean(fold_ap_list)),
        "fpr": fpr, "tpr": tpr, "roc_auc": roc_auc,    # pooled
        "prec": prec, "rec": rec, "pr_auc": pr_auc     # pooled
    }

# ────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("RandomForest baseline (tabular-only) — EXACT multimodal CV splits (no test set)")
    ecg_df = load_ecg_df(CSV_ECG)

    # Train two regimes with identical split policy to the multimodal script
    multi  = train_rf_variant(ecg_df, keep_multi=True,  name="MULTI stays")
    single = train_rf_variant(ecg_df, keep_multi=False, name="All stays")

    combined_dir = Path("rf_combined_performance")
    combined_dir.mkdir(exist_ok=True)

    # Save both pooled curves for comparison
    np.savez(combined_dir / "combined_roc_prc_data.npz",
             multi_fpr=multi["fpr"],  multi_tpr=multi["tpr"],  multi_auc=multi["roc_auc"],
             all_fpr=single["fpr"],   all_tpr=single["tpr"],   all_auc=single["roc_auc"],
             multi_prec=multi["prec"], multi_rec=multi["rec"], multi_pr=multi["pr_auc"],
             all_prec=single["prec"],  all_rec=single["rec"],  all_pr=single["pr_auc"])

    if HAVE_MPL:
        # Combined AUROC figure (pooled)
        plt.figure()
        plt.plot(multi["fpr"],  multi["tpr"],  label=f"MULTI stays (AUROC {multi['roc_auc']:.3f})")
        plt.plot(single["fpr"], single["tpr"], label=f"All stays (AUROC {single['roc_auc']:.3f})")
        plt.plot([0,1],[0,1], 'k--', alpha=.4)
        plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
        plt.title("RF CV ROC curves – MULTI vs All (pooled)"); plt.legend()
        plt.tight_layout(); plt.savefig(combined_dir / "combined_roc.png", dpi=200); plt.close()

        # Combined AUPRC figure (pooled)
        plt.figure()
        plt.plot(multi["rec"],  multi["prec"],  label=f"MULTI stays (AUPRC {multi['pr_auc']:.3f})")
        plt.plot(single["rec"], single["prec"], label=f"All stays (AUPRC {single['pr_auc']:.3f})")
        plt.xlabel("Recall"); plt.ylabel("Precision")
        plt.title("RF CV PR curves – MULTI vs All (pooled)"); plt.legend()
        plt.tight_layout(); plt.savefig(combined_dir / "combined_prc.png", dpi=200); plt.close()

    print("\nDone. Outputs:")
    print(f"  MULTI → {Path('rf_multi_performance').resolve()}")
    print(f"  ALL   → {Path('rf_all_performance').resolve()}")
    print(f"  COMBO → {combined_dir.resolve()}")
