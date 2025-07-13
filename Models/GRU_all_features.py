#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Train ED-LoS (> 6 h) predictor from sequential ECGs + sequential vitals
(+ per-stay scalar / boolean features).

Files
-----
sequential_ecgs.csv   one row per stay  (paths/times of up-to-6 ECGs + scalar cols)
vitals_long.csv       long format, one row per vital measurement
                      [stay_id, source, charttime,
                       temperature, heartrate, resprate, o2sat, sbp, dbp,
                       pain, sleeping, uta, real_pain …]
pretrained_model.pth  1-D ResNet-18 weights (12-lead ECG)

2025-06-13
"""
# ────────────────────────────────────────────────────────────────────────────
# Imports
# ────────────────────────────────────────────────────────────────────────────
import os, random, warnings, re
import numpy as np
import pandas as pd
from pathlib import Path       # ⬅ up top with the other imports

from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    confusion_matrix, precision_recall_fscore_support
)

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm, trange
import matplotlib.pyplot as plt

import wfdb, neurokit2 as nk
pd.DataFrame.pad = pd.DataFrame.ffill
pd.Series.pad    = pd.Series.ffill
from sklearn.metrics import roc_curve, precision_recall_curve

def roc_prc_arrays(y_true: np.ndarray, y_prob: np.ndarray):
    """Return (fpr, tpr, auc, precision, recall, auprc)."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = roc_auc_score(y_true, y_prob)
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)
    return fpr, tpr, roc_auc, prec, rec, pr_auc
# ────────────────────────────────────────────────────────────────────────────
# Global settings / hyper-parameters
# ────────────────────────────────────────────────────────────────────────────
CSV_ECG          = "final_ecgs.csv"
CSV_VITALS_LONG  = "vitals_long_cleaned.csv"
WAVE_DIR         = "..\\ecg"
ENC_CKPT         = "pretrained_model.pth"
PATIENCE = 5  
DEVICE           = "cuda" if torch.cuda.is_available() else "cpu"

MAX_ECG_SEQ      = 6          # last N ECGs kept
ECG_SEQ_LEN      = 5000       # samples per lead
ECG_EMB_DIM      = 512
SEQ_FEAT_DIM  = ECG_EMB_DIM + 1
SCAL_HID    = 256 
MAX_VITAL_SEQ    = 10         # last N vital rows kept
# vitals we keep *in this order*  →  VITAL_DIM = 9
numeric_vital_cols = ["temperature", "heartrate", "resprate",
                      "o2sat", "sbp", "dbp"]
pain_col = ["pain"]
# one-hot flags: 1 = value present in original row, 0 = it was NaN  
flag_cols = [f"real_{col}" for col in numeric_vital_cols]  # ← NEW

categorical_vital_cols = ["sleeping", "uta", "real_pain"] + flag_cols
vital_cols             = numeric_vital_cols + pain_col + categorical_vital_cols
VITAL_DIM = len(vital_cols) + 1 


# per-stay scalar features
NUM_SCALARS      = ["n_ecg", "pre_los", "acuity", "age", "n_pyxis_total", "hxETC2_00000000", "hxETC2_00001000", "hxETC2_00002000", "hxETC2_00003000", "hxETC2_00004000", "hxETC2_00005000", "hxETC2_00006000", "n_chronic_meds",
                    "n_pmh_bins", "n_prior_stays", "tropT_first", "tropT_peak", "lact_first", "lact_peak", "hgb_first", "hgb_peak",
                        "creat_first", "creat_peak",  "bnp_first", "bnp_peak", "k_first", "k_peak", "mg_first", "mg_peak",
                        "wbc_first", "wbc_peak", "crp_first", "crp_peak" 
                      ]          # z-scored
BOOL_SCALARS = [
    "gender_M",
    "arrival_transport_AMBULANCE",
    "arrival_transport_HELICOPTER",
    "arrival_transport_WALK IN",
    "Chest Pain",
    "Dyspnea",
    "Presyncope",
    "Syncope", "previous_home", "previous_else", "first_stay", "etc3_00000000", "etc3_00000100", "etc3_00000200", "etc3_00000300", "etc3_00000400", "etc3_00000500", "etc3_00000600", "etc3_00000700", "etc3_00000800", "etc3_00000900", "etc3_00001000", "etc3_00001200", "etc3_00002500", "etc3_00002600", "etc3_00002700", "etc3_00002800", "etc3_00003000", "etc3_00003100", "etc3_00003600", "etc3_00003700", "etc3_00003900", "etc3_00004500", "etc3_00004600", "etc3_00005600", "etc3_00005700", "etc3_00005800", "etc3_00005900", "etc3_00006000", "etc3_00006100", "etc3_00006200", "etc3_00006400", "etc2_00000000", "etc2_00001000", "etc2_00002000", "etc2_00003000", "etc2_00004000", "etc2_00005000", "etc2_00006000",
    "pmh_001", "pmh_002", "pmh_003", "pmh_004", "pmh_005", "pmh_006", "pmh_007", "pmh_008", "pmh_009", "pmh_010", "pmh_011", "pmh_012", "pmh_013", "pmh_014", "pmh_015", "pmh_016", "pmh_018", "pmh_019", "pmh_020", "pmh_022", "pmh_023", "pmh_025", "pmh_026", "pmh_027", "pmh_028", "pmh_029", "pmh_030", "pmh_031", "pmh_032", "pmh_033", "pmh_034", "pmh_035", "pmh_036", "pmh_037", "pmh_038", "pmh_039", "pmh_040", "pmh_041", "pmh_042", "pmh_044", "pmh_045", "pmh_047", "pmh_048", "pmh_049", "pmh_050", "pmh_051", "pmh_054", "pmh_055", "pmh_056", "pmh_057", "pmh_058", "pmh_059", "pmh_061", "pmh_062", "pmh_063", "pmh_064", "pmh_066", "pmh_067", "pmh_068", "pmh_069", "pmh_070", "pmh_072", "pmh_073", "pmh_074", "pmh_075", "pmh_076", "pmh_077", "pmh_078", "pmh_081", "pmh_082", "pmh_083", "pmh_084", "pmh_085", "pmh_086", "pmh_087", "pmh_088", "pmh_089", "pmh_090", "pmh_091", "pmh_092", "pmh_093", "pmh_094", "pmh_095", "pmh_096", "pmh_097", "pmh_099", "pmh_100", "pmh_101", "pmh_102", "pmh_103", "pmh_104", "pmh_105", "pmh_106", "pmh_107", "pmh_108", "pmh_109", "pmh_110", "pmh_113", "pmh_114", "pmh_115", "pmh_116", "pmh_117", "pmh_118", "pmh_121", "pmh_122", "pmh_124", "pmh_125", "pmh_126", "pmh_127", "pmh_128", "pmh_131", "pmh_133", "pmh_134", "pmh_136", "pmh_137", "pmh_138", "pmh_139", "pmh_140", "pmh_141", "pmh_143", "pmh_145", "pmh_146", "pmh_147", "pmh_148", "pmh_149", "pmh_151", "pmh_153", "pmh_155", "pmh_156", "pmh_159", "pmh_160", "pmh_161", "pmh_162", "pmh_163", "pmh_164", "pmh_165", "pmh_166", "pmh_167", "pmh_168", "pmh_170", "pmh_173", "pmh_175", "pmh_176", "pmh_177", "pmh_178", "pmh_180", "pmh_181", "pmh_182", "pmh_183", "pmh_184", "pmh_185", "pmh_186", "pmh_187", "pmh_188", "pmh_189", "pmh_190", "pmh_191", "pmh_192", "pmh_193", "pmh_195", "pmh_196", "pmh_197", "pmh_198", "pmh_199", "pmh_200", "pmh_201", "pmh_202", "pmh_203", "pmh_204", "pmh_205", "pmh_206", "pmh_207", "pmh_208", "pmh_209", "pmh_211", "pmh_212", "pmh_213", "pmh_214", "pmh_215", "pmh_216", "pmh_217", "pmh_224", "pmh_225", "pmh_226", "pmh_227", "pmh_228", "pmh_229", "pmh_230", "pmh_231", "pmh_232", "pmh_233", "pmh_234", "pmh_235", "pmh_236", "pmh_237", "pmh_238", "pmh_239", "pmh_240", "pmh_242", "pmh_243", "pmh_244", "pmh_248", "pmh_249", "pmh_250", "pmh_251", "pmh_252", "pmh_253", "pmh_254", "pmh_255", "pmh_256", "pmh_257", "pmh_258", "pmh_259", "pmh_2601", "pmh_2603", "pmh_2604", "pmh_2605", "pmh_2606", "pmh_2607", "pmh_2608", "pmh_2609", "pmh_2610", "pmh_2611", "pmh_2613", "pmh_2614", "pmh_2615", "pmh_2616", "pmh_2617", "pmh_2618", "pmh_2619", "pmh_2620", "pmh_2621", "pmh_BLD", "pmh_CIR", "pmh_DEN", "pmh_DIG", "pmh_EAR", "pmh_END", "pmh_EXT", "pmh_EYE", "pmh_FAC", "pmh_GEN", "pmh_INF", "pmh_INJ", "pmh_MAL", "pmh_MBD", "pmh_MUS", "pmh_NEO", "pmh_NVS", "pmh_PRG", "pmh_RSP", "pmh_SKN", "pmh_SYM",
    "tropT_abn", "tropT_missing", "lact_abn", "lact_missing", "hgb_abn", "hgb_missing", "creat_abn", "creat_missing",  "bnp_abn", "bnp_missing",
    "k_abn", "k_missing", "mg_abn", "mg_missing",  "wbc_abn", "wbc_missing", "crp_abn", "crp_missing"


]
                       # add your own extra booleans here
RECORD_LIST_CSV = "record_list.csv"          # path, ecg_time
rec_df   = pd.read_csv(RECORD_LIST_CSV, parse_dates=["ecg_time"])
PATH2TIME = dict(zip(rec_df["path"], rec_df["ecg_time"]))
HIDDEN           = 128
BATCH_SIZE       = 32
EPOCHS           = 20
 

LR_NET           = 3e-4
LR_ADAPTER       = 1e-4
LR_FINETUNE      = 5e-5

# silence NeuroKit “missing points” spam
warnings.filterwarnings(
    "ignore",
    message=r"There are .* missing data points in your signal.*",
    category=UserWarning,
    module=r".*neurokit2\.ecg\.ecg_clean.*"
)
# ── inspect a single, fully-pre-processed example ───────────────────────────
import json, pickle, pathlib


# --------------------------------------------------------------------------
# example usage – after you’ve built tr_ds / va_ds
  # set save=False if you only want a printout

# ────────────────────────────────────────────────────────────────────────────
# ECG loader  →  (12, ECG_SEQ_LEN) float32
# ────────────────────────────────────────────────────────────────────────────
def load_ecg(path: str) -> torch.Tensor:
    """
    Return a (12, ECG_SEQ_LEN) float32 tensor.
    If the file is unreadable or filtering fails, raise an Exception.
    """
    rec = wfdb.rdrecord(path)           # may raise FileNotFoundError, etc.
    sig = rec.p_signal.astype(float)    # (N, 12)

    # Replace non-finite values
    sig[~np.isfinite(sig)] = 0.0

    # Ensure each lead has non-zero variance
    flat = (np.abs(sig).sum(axis=0) == 0)
    if flat.any():
        sig[:, flat] += 1e-6 * np.random.randn(sig.shape[0], flat.sum())

    # Lead-wise clean
    cleaned = np.stack(
        [nk.ecg_clean(sig[:, j], sampling_rate=500, method="nk") for j in range(12)],
        axis=0,
    )

    # Trim / pad to fixed length
    if cleaned.shape[1] >= ECG_SEQ_LEN:
        cleaned = cleaned[:, :ECG_SEQ_LEN]
    else:
        cleaned = np.pad(cleaned, ((0, 0), (0, ECG_SEQ_LEN - cleaned.shape[1])))
    return torch.tensor(cleaned, dtype=torch.float32)

# ────────────────────────────────────────────────────────────────────────────
# 1-D ResNet-18 backbone  (unchanged from previous scripts)
# ────────────────────────────────────────────────────────────────────────────
class BasicBlock1D(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_planes, planes, 3, stride, 1, bias=False)
        self.bn1   = nn.BatchNorm1d(planes)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(planes, planes, 3, 1, 1, bias=False)
        self.bn2   = nn.BatchNorm1d(planes)
        self.down  = (
            nn.Sequential(
                nn.Conv1d(in_planes, planes, 1, stride, bias=False),
                nn.BatchNorm1d(planes))
            if stride != 1 or in_planes != planes else None
        )
    def forward(self, x):
        idn = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.down: idn = self.down(idn)
        return self.relu(out + idn)

class ResNet1D(nn.Module):
    def __init__(self, block, layers):
        super().__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv1d(12, 64, 7, 2, 3, bias=False)
        self.bn1   = nn.BatchNorm1d(64); self.relu = nn.ReLU(inplace=True)
        self.maxp  = nn.MaxPool1d(3, 2, 1)
        self.layer1 = self._make(block, 64,  layers[0])
        self.layer2 = self._make(block, 128, layers[1], 2)
        self.layer3 = self._make(block, 256, layers[2], 2)
        self.layer4 = self._make(block, 512, layers[3], 2)
        self.avgp   = nn.AdaptiveAvgPool1d(1)
    def _make(self, block, planes, blocks, stride=1):
        layers = []; strides = [stride] + [1]*(blocks-1)
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x))); x = self.maxp(x)
        x = self.layer1(x); x = self.layer2(x)
        x = self.layer3(x); x = self.layer4(x)
        return self.avgp(x).squeeze(-1)

def resnet18_backbone(): return ResNet1D(BasicBlock1D, [2, 2, 2, 2])

class ECGEncoder(nn.Module):
    def __init__(self, ckpt=ENC_CKPT, emb_dim=ECG_EMB_DIM):
        super().__init__()
        self.backbone = resnet18_backbone()
        sd = torch.load(ckpt, map_location="cpu")
        self.backbone.load_state_dict(
            {k: v for k, v in sd.items() if k in self.backbone.state_dict()
             and "fc" not in k},
            strict=False,
        )
        for p in self.backbone.parameters(): p.requires_grad = False
        self.adapter = nn.Linear(512, emb_dim)
    def forward(self, x):
        return self.adapter(self.backbone(x))




# ────────────────────────────────────────────────────────────────────────────
# Load CSVs
# ────────────────────────────────────────────────────────────────────────────
ecg_df = pd.read_csv(CSV_ECG, parse_dates=["ecg_time", 'intime'])

ecg_df["missing_acuity"] = ecg_df["acuity"].isna().astype(float)
BOOL_SCALARS.append("missing_acuity")

acuity_med = ecg_df["acuity"].median()
ecg_df["acuity"] = ecg_df["acuity"].fillna(acuity_med)

assert ecg_df[NUM_SCALARS].isna().sum().sum() == 0
#  vitals_long.csv already filtered to ≤ last ECG; triage has NaT charttime
vital_df = (
    pd.read_csv(CSV_VITALS_LONG)
)
for col in numeric_vital_cols:
    vital_df[f"real_{col}"] = vital_df[col].notna().astype(float)

# ────────────────────────────────────────────────────────────────────────────
#  Dataset
# ────────────────────────────────────────────────────────────────────────────
class StaySeqDS(Dataset):
    """
    One item →
        ecg_seq   [L_e, ECG_EMB_DIM]
        vit_seq   [L_v, VITAL_DIM]
        scalars   [len(NUM_SCALARS) + len(BOOL_SCALARS)]
        label     (float)
    """
    def __init__(self, stay_ids, imp=None, fit_imputer=False,
                 num_stats=None, fit_stats=False, vit_stats=None,
                 fit_vit_stats=False):
        self.stay_ids = stay_ids
        self.ecg_df   = ecg_df.set_index("stay_id")
        self.vital_df = vital_df
        self.bad = set()  
        # imputer (numerical vitals)
        if fit_imputer:
            self.imputer = IterativeImputer(random_state=0).fit(
                self.vital_df[numeric_vital_cols]
            )
        else:
            self.imputer = imp

        # z-score stats for numeric scalars
        if fit_stats:
            tr = self.ecg_df.loc[stay_ids, NUM_SCALARS]
            self.mean_ = tr.mean(); self.std_ = tr.std().replace(0, 1e-8)
        else:
            self.mean_, self.std_ = num_stats

        if fit_vit_stats:
            tr = self.vital_df[self.vital_df["stay_id"].isin(stay_ids)]
            mu = tr[numeric_vital_cols + pain_col].mean()
            sd = tr[numeric_vital_cols + pain_col].std().replace(0, 1e-8)
            self.vmean_, self.vstd_ = mu, sd
        else:
            self.vmean_, self.vstd_ = vit_stats
    # ─────────────────────────────────────────────
    def __len__(self): return len(self.stay_ids)

    # ─────────────────────────────────────────────
    def _ecg_seq(self, sid):
        row      = self.ecg_df.loc[sid]
        t_final  = row["ecg_time"]                # timestamp of last ECG in stay

        paths = [row.get(f"path_{i}") for i in range(1, MAX_ECG_SEQ)]
        paths = [p for p in paths if isinstance(p, str) and p] + [row["final_ecg_path"]]
        paths = paths[-MAX_ECG_SEQ:]

        sig_list, dt_list = [], []

        for p in paths:
            full = os.path.join(WAVE_DIR, p)
            try:
                sig_list.append(load_ecg(full))   # (12, 5000)
                t_ecg = PATH2TIME.get(p, pd.NaT)
                # Δt ≥ 0 in **hours** (earlier ECG → positive number)
                dt_hours = max((t_final - t_ecg).total_seconds() / 3600.0, 0.0)
                dt_list.append(dt_hours)
            except Exception as e:
                warnings.warn(f"ECG load failed [{sid}] {full}: {e}")

        if not sig_list:
            self.bad.add(sid); return None

        ecg_stack = torch.stack(sig_list)             # (L, 12, 5000)
        dt_tensor = torch.tensor(dt_list, dtype=torch.float32)  # (L,)
        return ecg_stack, dt_tensor


    # ─────────────────────────────────────────────
    def _vital_seq(self, sid):
        sub = (
            self.vital_df[self.vital_df["stay_id"] == sid]
            .sort_values("source")          # ← ONLY this column
            .tail(MAX_VITAL_SEQ)            # keep up-to-N most-recent rows
        )
        if sub.empty:
            return None
        
        t_final = self.ecg_df.loc[sid, "ecg_time"]
        # fill NaT charttimes with intime for that stay
        intime  = self.ecg_df.loc[sid, "intime"]
        ct      = pd.to_datetime(sub["charttime"]).fillna(intime)
        dt_hours = (t_final - ct).dt.total_seconds().clip(lower=0) / 3600.0
        dt_hours = dt_hours.to_numpy(dtype=np.float32).reshape(-1, 1)  # (L,1)
        # 1) impute numeric vitals
        num_imp = self.imputer.transform(sub[numeric_vital_cols])  # (L, 6)
        pain_raw = sub["pain"].to_numpy()[:, None]              # (L,1)
        pain_z   = (pain_raw - self.vmean_.pain) / self.vstd_.pain
        num_imp = (num_imp - self.vmean_[numeric_vital_cols].values) / self.vstd_[numeric_vital_cols].values
        assert not np.isnan(num_imp).any(), f"imputer produced NaN for {sid}"
        # 2) pull in the untouched columns
        cat_raw = sub[categorical_vital_cols].values.astype(float) # (L, 3)

        # 3) stitch back together in the same order
        vals = np.concatenate(
        [dt_hours,                  # ⬅ insert FIRST (or wherever you like)
         num_imp, pain_z, cat_raw],
        axis=1
    )      # (L, 9)
        assert not np.isnan(vals).any(), f"NaNs in vitals for stay {sid}"
        
        # z-score
       
        return torch.tensor(vals, dtype=torch.float32, device=DEVICE)

    # ─────────────────────────────────────────────
    def __getitem__(self, idx):
        sid     = self.stay_ids[idx]
        ecg_seq, dt_seq = self._ecg_seq(sid)  
        vit_seq = self._vital_seq(sid)

        if ecg_seq is None:
            raise ValueError(f"[Dataset] No ECG sequence for stay_id={sid!r}")
        if vit_seq is None:
            raise ValueError(f"[Dataset] No vitals for    stay_id={sid!r}")

        if ecg_seq is None or vit_seq is None:
            return None

        #  per-stay scalars  (numeric z-scored, booleans as 0/1)
        num = ((self.ecg_df.loc[sid, NUM_SCALARS] - self.mean_) / self.std_).values
        boo = self.ecg_df.loc[sid, BOOL_SCALARS].astype(float).values if BOOL_SCALARS else []
        scalars = torch.tensor(
            np.concatenate([num, boo]).astype(np.float32),
            device=DEVICE,
        )

        label_val = float(self.ecg_df.loc[sid, "disposition_HOME"])  # python float
        label     = torch.tensor(label_val, dtype=torch.float32)      # still on CPU
        return ecg_seq, dt_seq, vit_seq, scalars, label              # no second wrap

# ────────────────────────────────────────────────────────────────────────────
def collate(batch):
    batch = [b for b in batch if b]
    if not batch:
        return None

    ecg, dt, vit, scal, y = zip(*batch)      #  ← NO lens here
    elens = torch.tensor([t.size(0) for t in ecg], device=DEVICE)
    vlens = torch.tensor([t.size(0) for t in vit], device=DEVICE)

    return (
        pad_sequence(ecg, batch_first=True),        # (B,T,12,5000)
        pad_sequence(dt,  batch_first=True),        # (B,T)
        elens,
        pad_sequence(vit, batch_first=True),
        vlens,
        torch.stack(scal),
        torch.stack(y).to(DEVICE),
    )



# ────────────────────────────────────────────────────────────────────────────
# Model
# ────────────────────────────────────────────────────────────────────────────
class LoSNet(nn.Module):
    def __init__(self, scal_dim):
        super().__init__()
        self.ecg_gru = nn.GRU(SEQ_FEAT_DIM, HIDDEN, batch_first=True)
        self.vit_gru = nn.GRU(VITAL_DIM,    HIDDEN, batch_first=True)
        self.scal_fc = nn.Linear(scal_dim, SCAL_HID)
        self.drop    = nn.Dropout(0.3)
        self.head    = nn.Linear(HIDDEN*2 + SCAL_HID, 1)

    def _last_h(self, gru, x, lens):
        packed = pack_padded_sequence(
            x, lens.cpu(), batch_first=True, enforce_sorted=False
        )
        _, h = gru(packed)
        return h.squeeze(0)

    def forward(self, ecg, elens, vit, vlens, scal):
        h_ecg = self._last_h(self.ecg_gru, ecg, elens)
        h_vit = self._last_h(self.vit_gru, vit, vlens)
        s     = torch.relu(self.scal_fc(scal))
        x     = torch.cat([h_ecg, h_vit, s], 1)
        return self.head(self.drop(x)).squeeze(1)

def inspect_random_example(ds: StaySeqDS, save=False, out_dir="example_dump"):
    sid = random.choice(ds.stay_ids)

    # unpack the 5-tuple returned by __getitem__
    ecg_seq, dt_seq, vit_seq, scalars, label = ds[ds.stay_ids.index(sid)]

    ecg_np  = ecg_seq.cpu().numpy()
    dt_np   = dt_seq.cpu().numpy()
    vit_np  = vit_seq.cpu().numpy()
    scal_np = scalars.cpu().numpy()

    print("\n===  RANDOM STAY  =================================================")
    print(f"stay_id                 : {sid}")
    print(f"label (>6 h LOS?)       : {bool(label.item())}")
    print(f"ecg_seq shape           : {ecg_np.shape}")            # (L_e,12,5000)
    print(f"dt_seq (h)              : {dt_np}")                   # new line
    print(f"vit_seq shape           : {vit_np.shape}")            # (L_v,{VITAL_DIM})
    print(f"scalars shape           : {scal_np.shape}")
    print("first ECG Δt (h)        :", dt_np[0])
    print("first ECG emb (trunc)   :", ecg_np[0, :, :10])
    print("first vitals row        :", vit_np[-1])
    print("first 10 scalar values  :", scal_np[:10])
    print("===================================================================")

    if save:
        out = Path(out_dir); out.mkdir(exist_ok=True)
        np.save(out / f"{sid}_ecg.npy",     ecg_np)
        np.save(out / f"{sid}_dt.npy",      dt_np)
        np.save(out / f"{sid}_vitals.npy",  vit_np)
        np.save(out / f"{sid}_scalars.npy", scal_np)
        with open(out / f"{sid}_meta.json", "w") as f:
            json.dump({"stay_id": sid, "label": bool(label.item())}, f, indent=2)
        print(f"✓ dumped tensors to {out.resolve()}")

# ────────────────────────────────────────────────────────────────────────────
# Train / eval helpers
# ────────────────────────────────────────────────────────────────────────────


def epoch_pass(net, loader, criterion, opt=None, desc="train"):
    train = opt is not None
    net.train()      if train else net.eval()
    encoder.train()  if train else encoder.eval()   # BN layers behave correctly

    yt, yp, losses = [], [], []
    bar = tqdm(loader, desc=desc, leave=False)

    for ecg_raw, dt, elens, vit, vlens, scal, y in bar:
        ecg_raw = ecg_raw.to(DEVICE)
        dt      = dt.to(DEVICE).unsqueeze(-1)      # (B,T,1)
        y       = y.to(DEVICE)

        B, T, C, L = ecg_raw.shape
        ecg_flat   = ecg_raw.view(B*T, C, L)
        emb_flat   = encoder(ecg_flat)             # (B*T, 512)
        emb_seq    = emb_flat.view(B, T, -1)       # (B,T,512)

        emb_seq = torch.cat([emb_seq, dt], dim=-1) #  (B,T,513)  ← Δt appended


        with torch.set_grad_enabled(train):
            logit = net(emb_seq, elens.to(DEVICE), vit, vlens, scal.to(DEVICE))
            loss  = criterion(logit, y)
            if train:
                opt.zero_grad()
                loss.backward()
                opt.step()

        yt.append(y.detach().cpu())
        yp.append(torch.sigmoid(logit).detach().cpu())
        losses.append(loss.item())
        bar.set_postfix(loss=f"{loss.item():.3f}")

    yt = torch.cat(yt).numpy()
    yp = torch.cat(yp).numpy()

    return (
        np.mean(losses),
        roc_auc_score(yt, yp),
        average_precision_score(yt, yp),
        yt,
        yp,
    )

# ────────────────────────────────────────────────────────────────────────────
# Main training loop
# ────────────────────────────────────────────────────────────────────────────
def train_variant(name="MULTI stays", keep_multi=True):
    global encoder
    encoder = ECGEncoder().to(DEVICE)
    encoder.eval()  
    for p in encoder.backbone.layer4.parameters():   # fine-tune last block
        p.requires_grad = True
    stays = ecg_df.loc[
        ecg_df["n_ecg"].ge(2) if keep_multi else slice(None), "stay_id"
    ].tolist()
    perf_dir  = Path("multi_performance" if keep_multi else "all_performance")
    perf_dir.mkdir(exist_ok=True)
    random.shuffle(stays)
    split = int(0.9 * len(stays))
    tr_ids, va_ids = stays[:split], stays[split:]
    tr_df  = ecg_df[ecg_df["stay_id"].isin(tr_ids)]
    labels = tr_df["disposition_HOME"].values       # 1 = positive
    n_pos  = labels.sum()
    n_neg  = len(labels) - n_pos
    pos_weight_tensor = torch.tensor(
        1.0 if n_pos == 0 else n_neg / n_pos,
        device=DEVICE
)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    tqdm.write(f"[{name}] pos_weight = {pos_weight_tensor.item():.2f}")
    tr_ds = StaySeqDS(tr_ids, fit_imputer=True, fit_stats=True, fit_vit_stats=True)
    va_ds = StaySeqDS(va_ids,
                  imp=tr_ds.imputer,
                  num_stats=(tr_ds.mean_, tr_ds.std_),
                  vit_stats=(tr_ds.vmean_, tr_ds.vstd_))
    inspect_random_example(tr_ds, save=False) 
    print(f"[INFO]  Unusable ECG stays skipped:"
      f"  train={len(tr_ds.bad)}   val={len(va_ds.bad)}")
    tr_dl = DataLoader(tr_ds, BATCH_SIZE, True,  collate_fn=collate)
    va_dl = DataLoader(va_ds, BATCH_SIZE, False, collate_fn=collate)

    net = LoSNet(scal_dim=len(NUM_SCALARS)+len(BOOL_SCALARS)).to(DEVICE)
    
    opt = torch.optim.AdamW(
        [{"params": net.parameters(),                         "lr": LR_NET},
         {"params": encoder.adapter.parameters(),             "lr": LR_ADAPTER},
         {"params": encoder.backbone.layer4.parameters(),     "lr": LR_FINETUNE}],
        weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.9)
    stem = name.lower().replace(" ", "_")
    history, best_auc = [], -np.inf             # ← keep all epoch stats here
    best_snapshot  = None             # will hold yt / yp etc. for BEST epoch
    best_row       = None    
    no_improve = 0
    for ep in trange(1, EPOCHS + 1, desc=f"{name} epochs"):
        tr_loss, tr_auc, _, _, _   = epoch_pass(net, tr_dl, criterion, opt,  "train")
        va_loss, va_auc, va_ap, yt, yp = epoch_pass(net, va_dl, criterion, None, "valid")
        scheduler.step()
        if va_auc > best_auc:
            best_auc = va_auc
            no_improve = 0
            torch.save(
                {
                    "epoch":   ep,
                    "val_auc": va_auc,
                    "val_ap":  va_ap,
                    "model":   net.state_dict(),      # LoSNet weights
                    "encoder": encoder.state_dict(),  # ECG-ResNet weights
                    # (optional) anything else you’d like
                },
                perf_dir / f"{stem}_best_model.pth"              # e.g. multi_stays_best_model.pth
            )
            best_snapshot = dict(yt=yt, yp=yp, val_auc=va_auc, val_ap=va_ap)
            best_row = {"epoch":   ep,
                "train_loss": tr_loss,
                "val_loss":   va_loss,
                "train_auc":  tr_auc,
                "val_auc":    va_auc,
                "val_ap":     va_ap}
        else:
            no_improve += 1
        tqdm.write(f"[{name}] ep {ep:02d} │ "
                   f"train loss {tr_loss:.4f} AUC {tr_auc:.3f} │ "
                   f"val loss {va_loss:.4f} AUC {va_auc:.3f}")

        history.append({"epoch": ep,
                        "train_loss": tr_loss,
                        "val_loss":   va_loss,
                        "train_auc":  tr_auc,
                        "val_auc":    va_auc,
                        "val_ap":     va_ap})
        if no_improve >= PATIENCE:
            tqdm.write(f"↳ Early stopping at epoch {ep} "
                    f"(no val AUC ↑ for {PATIENCE} epochs)")
            break
    yt = best_snapshot["yt"]          # numpy arrays
    yp = best_snapshot["yp"]
    # ── confusion matrix on final validation predictions ────────────────────
    pred = (yp > 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(yt, pred).ravel()
    pr, rc, f1, _ = precision_recall_fscore_support(
        yt, pred, average="binary", zero_division=0)

    # history CSV: keep all epochs *plus* flag the best one
    metrics_df = pd.DataFrame(history)
    metrics_df["is_best"] = metrics_df["epoch"] == best_row["epoch"]
    metrics_df.to_csv(perf_dir / f"{stem}_metrics.csv", index=False)

    pd.DataFrame([{
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "precision": pr, "recall": rc, "f1": f1,
        "val_auc": best_snapshot["val_auc"],            # ★ best epoch
        "val_ap":  best_snapshot["val_ap"]              # ★ best epoch
    }]).to_csv(perf_dir / f"{stem}_confusion.csv", index=False)

    # 3) loss-curve plot  (unchanged: still every epoch)  ──────────────
    plt.figure(figsize=(5, 3))
    epochs = [h["epoch"] for h in history]
    plt.plot(epochs, [h["train_loss"] for h in history], label="train")
    plt.plot(epochs, [h["val_loss"]   for h in history], label="val")
    plt.xlabel("epoch"); plt.ylabel("BCE loss"); plt.title("Loss curve")
    plt.legend(); plt.tight_layout()
    plt.savefig(perf_dir / f"{stem}_loss.png", dpi=200)
    plt.close()

    fpr, tpr, roc_auc, prec, rec, pr_auc = roc_prc_arrays(yt, yp)
    np.savez(perf_dir / f"{stem}_roc_prc_data.npz",
            fpr=fpr, tpr=tpr, roc_auc=roc_auc,
            prec=prec, rec=rec, pr_auc=pr_auc)
    return {
            "name": name,
            "fpr":  fpr,
            "tpr":  tpr,
            "roc_auc": roc_auc,
            "prec": prec,
            "rec":  rec,
            "pr_auc": pr_auc,
        }

    
# ────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    random.seed(0); np.random.seed(0); torch.manual_seed(0)
    multi  = train_variant("MULTI stays", keep_multi=True)
    single = train_variant("All stays",  keep_multi=False)
    combined_dir = Path("combined_performance")
    combined_dir.mkdir(exist_ok=True)

    np.savez(combined_dir / "combined_roc_prc_data.npz",
        multi_fpr=multi["fpr"],  multi_tpr=multi["tpr"],  multi_auc=multi["roc_auc"],
        all_fpr=single["fpr"],   all_tpr=single["tpr"],   all_auc=single["roc_auc"],
        multi_prec=multi["prec"], multi_rec=multi["rec"], multi_pr=multi["pr_auc"],
        all_prec=single["prec"],  all_rec=single["rec"],  all_pr=single["pr_auc"])

    # ── combined AUROC figure ──────────────────────────────────────
    plt.figure()
    plt.plot(multi["fpr"],  multi["tpr"],
             label=f"MULTI stays (AUC {multi['roc_auc']:.3f})")
    plt.plot(single["fpr"], single["tpr"],
             label=f"All stays (AUC {single['roc_auc']:.3f})")
    plt.plot([0,1],[0,1], 'k--', alpha=.4)
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("ROC curves – MULTI vs All stays"); plt.legend()
    plt.tight_layout(); plt.savefig(combined_dir / "combined_roc.png", dpi=200); plt.close()

    # ── combined AUPRC figure ─────────────────────────────────────
    plt.figure()
    plt.plot(multi["rec"],  multi["prec"],
             label=f"MULTI stays (AUPRC {multi['pr_auc']:.3f})")
    plt.plot(single["rec"], single["prec"],
             label=f"All stays (AUPRC {single['pr_auc']:.3f})")
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title("PR curves – MULTI vs All stays"); plt.legend()
    plt.tight_layout(); plt.savefig(combined_dir / "combined_prc.png", dpi=200); plt.close()
