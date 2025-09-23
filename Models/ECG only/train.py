#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ECG-only ED-disposition model:
  • MULTI stays (n_ecg >= 2)  and  All stays (all)
  • 5-fold CV only (NO held-out test set)
  • EXACT same split policy as multimodal: StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
  • Saves OOF predictions for DeLong, per-fold metrics, loss curves, and pooled ROC/PR

Outputs (per variant):
  ecg_only_performance/{multi_performance | all_performance}/
    - {stem}_fold{k}_metrics.csv
    - {stem}_fold{k}_loss.png
    - {stem}_val_predictions.csv      (OOF preds: stay_id, true_label, pred_prob, fold)
    - {stem}_cv_summary.csv           (mean±std across folds + pooled AUROC/AUPRC)
    - {stem}_cv_roc.png               (pooled ROC)
    - {stem}_cv_prc.png               (pooled PR)
  ecg_only_performance/combined_performance/
    - combined_roc_prc_data.npz
    - combined_roc.png
    - combined_prc.png
"""

# ───────────────────────── Imports ──────────────────────────
import os, random, warnings, sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm, trange
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from torch.utils.data import Dataset, DataLoader

import wfdb
import neurokit2 as nk
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    roc_curve, precision_recall_curve
)
from sklearn.model_selection import StratifiedKFold

# ─────────────────── Global hyper-parameters ───────────────────
CSV_ECG   = "final_ecgs.csv"
WAVE_DIR  = "..\\ecg"
ENC_CKPT  = "pretrained_model.pth"
OUT_ROOT  = Path("ecg_only_performance")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MAX_ECG_SEQ   = 6      # keep last N ECGs
ECG_SEQ_LEN   = 5_000  # samples per lead
ECG_EMB_DIM   = 512    # output of ResNet18+adapter
SEQ_FEAT_DIM  = ECG_EMB_DIM + 1
HIDDEN        = 256
BATCH_SIZE    = 32
EPOCHS        = 20
PATIENCE_ES   = 3      # early stopping within each fold
LR_NET        = 3e-4
base_lr       = 1e-4
LR_GAMMA      = 0.9
SEED_SPLITS   = 0      # must be 0 to match multimodal script’s folds
SEED_TRAIN    = 10

# reproducibility
def set_seed(s=SEED_TRAIN):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)

pd.DataFrame.pad = pd.DataFrame.ffill
pd.Series.pad    = pd.Series.ffill

warnings.filterwarnings(
    "ignore",
    message=r"There are .* missing data points in your signal.*",
    category=UserWarning,
    module=r".*neurokit2\.ecg\.ecg_clean.*",
)

# ─────────────────────── Load CSVs / helpers ───────────────────────
ecg_df = pd.read_csv(CSV_ECG, parse_dates=["ecg_time"])
record_df = pd.read_csv("record_list.csv", parse_dates=["ecg_time"])
PATH2TIME = dict(zip(record_df["path"], record_df["ecg_time"]))

def roc_prc_arrays(y_true: np.ndarray, y_prob: np.ndarray):
    """Return fpr, tpr, auc, prec, rec, auprc."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc  = roc_auc_score(y_true, y_prob)
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    ap   = average_precision_score(y_true, y_prob)
    return fpr, tpr, auc, prec, rec, ap

# ─────────────────────── ECG loader ───────────────────────────
def load_ecg(path: str) -> torch.Tensor:
    """Return a (12, ECG_SEQ_LEN) float32 tensor from a raw WFDB file."""
    rec = wfdb.rdrecord(path)
    sig = rec.p_signal.astype(float)
    sig[~np.isfinite(sig)] = 0.0
    flat = (np.abs(sig).sum(axis=0) == 0)
    if flat.any():
        sig[:, flat] += 1e-6 * np.random.randn(sig.shape[0], flat.sum())
    cleaned = np.stack(
        [nk.ecg_clean(sig[:, j], sampling_rate=500, method="nk") for j in range(12)],
        axis=0,
    )
    if cleaned.shape[1] >= ECG_SEQ_LEN:
        cleaned = cleaned[:, :ECG_SEQ_LEN]
    else:
        cleaned = np.pad(cleaned, ((0, 0), (0, ECG_SEQ_LEN - cleaned.shape[1])))
    return torch.tensor(cleaned, dtype=torch.float32)

# ─────────────────────── ECG encoder ──────────────────────────
class BasicBlock1D(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_planes, planes, 3, stride, 1, bias=False)
        self.bn1   = nn.BatchNorm1d(planes)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(planes, planes, 3, 1, 1, bias=False)
        self.bn2   = nn.BatchNorm1d(planes)
        self.down  = (nn.Sequential(nn.Conv1d(in_planes, planes, 1, stride, bias=False),
                                    nn.BatchNorm1d(planes))
                      if stride != 1 or in_planes != planes else None)
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
        self.bn1   = nn.BatchNorm1d(64)
        self.relu  = nn.ReLU(inplace=True)
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
        x = self.layer1(x); x = self.layer2(x); x = self.layer3(x); x = self.layer4(x)
        return self.avgp(x).squeeze(-1)

def resnet18_backbone(): return ResNet1D(BasicBlock1D, [2, 2, 2, 2])

class ECGEncoder(nn.Module):
    """Pretrained ResNet18 backbone with a linear adapter to produce ECG embeddings."""
    def __init__(self, ckpt: str = ENC_CKPT, emb_dim: int = ECG_EMB_DIM):
        super().__init__()
        self.backbone = resnet18_backbone()
        sd = torch.load(ckpt, map_location="cpu")
        self.backbone.load_state_dict(
            {k: v for k, v in sd.items() if k in self.backbone.state_dict() and "fc" not in k},
            strict=False,
        )
        for p in self.backbone.parameters(): p.requires_grad = False
        self.adapter = nn.Linear(512, emb_dim)
        torch.nn.init.kaiming_normal_(self.adapter.weight, nonlinearity="relu")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.adapter(self.backbone(x))

# ─────────────────────── Dataset ─────────────────────────────
class StayECG(Dataset):
    """Returns ECG sequences, delta-time channels, and labels for a list of stays."""
    def __init__(self, stay_ids):
        self.stay_ids = stay_ids
        self.ecg_df   = ecg_df.set_index("stay_id")
    def __len__(self): return len(self.stay_ids)
    def __getitem__(self, idx: int):
        sid = self.stay_ids[idx]
        row = self.ecg_df.loc[sid]
        t_final = row["ecg_time"]
        # collect up to MAX_ECG_SEQ ECGs ending with final_ecg_path
        paths = [row.get(f"path_{i}") for i in range(1, MAX_ECG_SEQ)]
        paths = [p for p in paths if isinstance(p, str) and p] + [row["final_ecg_path"]]
        paths = paths[-MAX_ECG_SEQ:]
        sig_list, dt_list = [], []
        for p in paths:
            full = os.path.join(WAVE_DIR, p)
            sig  = load_ecg(full)
            sig_list.append(sig)
            t = PATH2TIME.get(p, pd.NaT)
            dt_hours = max((t_final - t).total_seconds() / 3600.0, 0.0)
            dt_list.append(dt_hours)
        ecg_seq = torch.stack(sig_list)                       # (L, 12, 5000)
        dt_seq  = torch.tensor(dt_list, dtype=torch.float32)  # (L,)
        label   = torch.tensor(float(row["disposition_HOME"]), dtype=torch.float32)
        return ecg_seq, dt_seq, label

def collate(batch):
    ecg, dt, y = zip(*batch)
    lens = torch.tensor([t.size(0) for t in ecg])
    y    = torch.stack(y)
    ecg  = pad_sequence(ecg, batch_first=True)  # (B, T, 12, 5000)
    dt   = pad_sequence(dt,  batch_first=True)  # (B, T)
    return ecg, dt, lens, y

# ─────────────────── Model definitions ───────────────────────
class GRUHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru  = nn.GRU(SEQ_FEAT_DIM, HIDDEN, batch_first=True)
        self.drop = nn.Dropout(0.3)
        self.out  = nn.Linear(HIDDEN, 1)
    def _last(self, x: torch.Tensor, lens: torch.Tensor) -> torch.Tensor:
        packed = pack_padded_sequence(x, lens.cpu(), batch_first=True, enforce_sorted=False)
        _, h = self.gru(packed)
        return h.squeeze(0)
    def forward(self, x: torch.Tensor, lens: torch.Tensor) -> torch.Tensor:
        h = self._last(x, lens)
        return self.out(self.drop(h)).squeeze(1)

HEADS = {"gru": GRUHead}

# ─────────────── Train / eval helpers ────────────────────────
def epoch_pass(head: nn.Module, loader: DataLoader, criterion, opt=None, desc="train", encoder=None):
    train = opt is not None
    head.train()     if train else head.eval()
    encoder.train()  if train else encoder.eval()

    yt, yp, losses = [], [], []
    bar = tqdm(loader, desc=desc, leave=False)
    for ecg, dt, lens, y in bar:
        ecg = ecg.to(DEVICE)  # (B,T,12,5000)
        y   = y.to(DEVICE)
        B, T, C, L = ecg.shape
        ecg_flat = ecg.view(B*T, C, L)
        emb_flat = encoder(ecg_flat)                 # (B*T,512)
        emb      = emb_flat.view(B, T, -1)           # (B,T,512)
        dt       = dt.to(DEVICE).unsqueeze(-1)       # (B,T,1)
        x        = torch.cat([emb, dt], dim=-1)      # (B,T,513)

        with torch.set_grad_enabled(train):
            logit = head(x, lens.to(DEVICE))
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
    return np.mean(losses), roc_auc_score(yt, yp), average_precision_score(yt, yp), yt, yp

# ──────────────── main CV per regime ────────────────
def train_variant(head_cls, name: str, keep_multi: bool, perf_dir: Path):
    """
    Run 5-fold CV with EXACT same fold generator as multimodal script.
    Saves OOF preds, per-fold metrics/loss, pooled CV summary and curves.
    """
    # Select stays per regime
    stays = ecg_df.loc[ecg_df["n_ecg"].ge(2) if keep_multi else slice(None), "stay_id"].tolist()
    y_all = ecg_df.set_index("stay_id").loc[stays, "disposition_HOME"].to_numpy()

    # StratifiedKFold with random_state=0 to match multimodal script
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED_SPLITS)
    folds = list(skf.split(stays, y_all))

    perf_dir.mkdir(parents=True, exist_ok=True)
    stem = name.lower().replace(" ", "_")

    all_val_preds = []
    fold_summaries = []

    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        set_seed(SEED_TRAIN)

        fold_tr_ids = [stays[i] for i in train_idx]
        fold_va_ids = [stays[i] for i in val_idx]

        # Fold-specific class weight (match multimodal style)
        labels_tr = ecg_df.set_index("stay_id").loc[fold_tr_ids, "disposition_HOME"].values
        n_pos = labels_tr.sum(); n_neg = len(labels_tr) - n_pos
        pos_w = 1.0 if n_pos == 0 else n_neg / max(n_pos, 1)
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_w, device=DEVICE))
        tqdm.write(f"[{name}] Fold {fold_idx+1} pos_weight = {pos_w:.2f}")

        # Data
        tr_ds = StayECG(fold_tr_ids)
        va_ds = StayECG(fold_va_ids)
        tr_dl = DataLoader(tr_ds, BATCH_SIZE, True,  collate_fn=collate)
        va_dl = DataLoader(va_ds, BATCH_SIZE, False, collate_fn=collate)

        # Model
        encoder = ECGEncoder().to(DEVICE)
        # unfreeze highest block for light finetuning
        for p in encoder.backbone.layer4.parameters(): p.requires_grad = True
        head = head_cls().to(DEVICE)

        opt = torch.optim.AdamW(
            [{"params": head.parameters(),                      "lr": LR_NET},
             {"params": encoder.adapter.parameters(),           "lr": LR_NET},
             {"params": encoder.backbone.layer4.parameters(),   "lr": base_lr}],
            weight_decay=1e-4
        )
        sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=LR_GAMMA)

        best_auc = -np.inf
        best_snapshot = None
        history = []
        no_improve = 0

        for ep in trange(1, EPOCHS + 1, desc=f"{name} fold {fold_idx+1}"):
            tr_loss, tr_auc, _, _, _ = epoch_pass(head, tr_dl, criterion, opt, "train", encoder)
            va_loss, va_auc, va_ap, yt, yp = epoch_pass(head, va_dl, criterion, None, "valid", encoder)
            sched.step()

            tqdm.write(f"[{name}] Fold {fold_idx+1} │ ep {ep:02d} │ train loss {tr_loss:.4f} AUC {tr_auc:.3f} │ val loss {va_loss:.4f} AUC {va_auc:.3f}")
            history.append({"epoch": ep, "train_loss": tr_loss, "val_loss": va_loss,
                            "train_auc": tr_auc, "val_auc": va_auc, "val_ap": va_ap})

            if va_auc > best_auc:
                best_auc = va_auc; no_improve = 0
                best_snapshot = dict(yt=yt, yp=yp, val_auc=va_auc, val_ap=va_ap,
                                     epoch=ep, tr_loss=tr_loss, va_loss=va_loss, tr_auc=tr_auc,
                                     fold=fold_idx, stay_ids=fold_va_ids)
                torch.save({"model": head.state_dict(), "encoder": encoder.state_dict()},
                           perf_dir / f"{stem}_fold{fold_idx}_best_model.pth")
            else:
                no_improve += 1

            if no_improve >= PATIENCE_ES:
                tqdm.write(f"↳ Early stopping at epoch {ep} (no val AUC ↑ for {PATIENCE_ES} epochs)")
                break

        # Per-fold metrics & loss curve
        mdf = pd.DataFrame(history)
        mdf["is_best"] = mdf["epoch"] == best_snapshot["epoch"]
        mdf.to_csv(perf_dir / f"{stem}_fold{fold_idx}_metrics.csv", index=False)

        plt.figure(figsize=(5,3))
        epochs = mdf["epoch"].tolist()
        plt.plot(epochs, mdf["train_loss"], label="train")
        plt.plot(epochs, mdf["val_loss"],   label="val")
        plt.xlabel("epoch"); plt.ylabel("BCE loss"); plt.title(f"Fold {fold_idx+1} Loss")
        plt.legend(); plt.tight_layout()
        plt.savefig(perf_dir / f"{stem}_fold{fold_idx}_loss.png", dpi=200); plt.close()

        # Collect OOF predictions
        val_df = pd.DataFrame({
            "stay_id": best_snapshot["stay_ids"],
            "true_label": best_snapshot["yt"],
            "pred_prob":  best_snapshot["yp"],
            "fold": fold_idx
        })
        all_val_preds.append(val_df)
        fold_summaries.append(best_snapshot)

    # Aggregate OOF predictions (for pooled metrics / DeLong later)
    oof_df = pd.concat(all_val_preds, ignore_index=True)
    oof_df.to_csv(perf_dir / f"{stem}_val_predictions.csv", index=False)

    yt_all = oof_df["true_label"].to_numpy()
    yp_all = oof_df["pred_prob"].to_numpy()
    fpr, tpr, auc, prec, rec, ap = roc_prc_arrays(yt_all, yp_all)

    # CV summary: mean±std across folds (best checkpoints) + pooled metrics
    aucs = [m["val_auc"] for m in fold_summaries]
    aps  = [m["val_ap"]  for m in fold_summaries]
    summary = pd.DataFrame({
        "metric": ["val_auc_mean","val_auc_std","val_ap_mean","val_ap_std","pooled_auc","pooled_ap","n_samples"],
        "value":  [np.mean(aucs), np.std(aucs, ddof=1), np.mean(aps), np.std(aps, ddof=1), auc, ap, len(yt_all)]
    })
    summary.to_csv(perf_dir / f"{stem}_cv_summary.csv", index=False)

    # Save pooled curves + figures
    np.savez(perf_dir / f"{stem}_cv_roc_prc_data.npz", fpr=fpr, tpr=tpr, auc=auc, prec=prec, rec=rec, pr=ap)

    plt.figure()
    plt.plot(fpr, tpr, label=f"{name} (pooled AUROC {auc:.3f})")
    plt.plot([0,1],[0,1],'k--', alpha=.4)
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title(f"{name} – CV ROC (pooled)"); plt.legend()
    plt.tight_layout(); plt.savefig(perf_dir / f"{stem}_cv_roc.png", dpi=200); plt.close()

    plt.figure()
    plt.plot(rec, prec, label=f"{name} (pooled AUPRC {ap:.3f})")
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title(f"{name} – CV PR (pooled)"); plt.legend()
    plt.tight_layout(); plt.savefig(perf_dir / f"{stem}_cv_prc.png", dpi=200); plt.close()

    return {"name": name, "fpr": fpr, "tpr": tpr, "auc": auc, "prec": prec, "rec": rec, "ap": ap}

# ───────────────────────── main ─────────────────────────────
def main(arch="gru"):
    assert arch in HEADS, f"Unknown architecture '{arch}'. Choose from {list(HEADS)}"
    head_cls = HEADS[arch]

    # MULTI
    multi_dir = OUT_ROOT / "multi_performance"
    multi = train_variant(head_cls, "MULTI stays", keep_multi=True,  perf_dir=multi_dir)

    # ALL
    all_dir = OUT_ROOT / "all_performance"
    single = train_variant(head_cls, "All stays",  keep_multi=False, perf_dir=all_dir)

    # Combined plots (pooled)
    comb_dir = OUT_ROOT / "combined_performance"
    comb_dir.mkdir(parents=True, exist_ok=True)
    np.savez(comb_dir / "combined_roc_prc_data.npz",
             multi_fpr=multi["fpr"],  multi_tpr=multi["tpr"],  multi_auc=multi["auc"],
             all_fpr=single["fpr"],   all_tpr=single["tpr"],   all_auc=single["auc"],
             multi_prec=multi["prec"], multi_rec=multi["rec"], multi_pr=multi["ap"],
             all_prec=single["prec"],  all_rec=single["rec"],  all_pr=single["ap"])

    plt.figure()
    plt.plot(multi["fpr"],  multi["tpr"],  label=f"MULTI stays (AUROC {multi['auc']:.3f})")
    plt.plot(single["fpr"], single["tpr"], label=f"All stays (AUROC {single['auc']:.3f})")
    plt.plot([0,1],[0,1],'k--', alpha=.4)
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("CV ROC curves – MULTI vs All (pooled)"); plt.legend()
    plt.tight_layout(); plt.savefig(comb_dir / "combined_roc.png", dpi=200); plt.close()

    plt.figure()
    plt.plot(multi["rec"],  multi["prec"],  label=f"MULTI stays (AUPRC {multi['ap']:.3f})")
    plt.plot(single["rec"], single["prec"], label=f"All stays (AUPRC {single['ap']:.3f})")
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title("CV PR curves – MULTI vs All (pooled)"); plt.legend()
    plt.tight_layout(); plt.savefig(comb_dir / "combined_prc.png", dpi=200); plt.close()

if __name__ == "__main__":
    # folds must use SEED_SPLITS=0 for exact match with multimodal
    random.seed(SEED_SPLITS); np.random.seed(SEED_SPLITS); torch.manual_seed(SEED_SPLITS)
    arch = "gru" if len(sys.argv) == 1 else sys.argv[1].lower()
    main(arch)
