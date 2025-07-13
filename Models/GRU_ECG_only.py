#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Train ED‑LoS (>6 h) predictor from *only* sequential ECGs.
Three architectures are supported:
    1. GRU            ("gru") – similar to previous script but vitals/scalars removed
    2. Transformer    ("transformer") – 2‑layer encoder with learnable positional enc.
    3. Temporal CNN   ("tcn") – dilated 1‑D convolutions (Bai et al., 2018).

For each architecture the script trains two data regimes
    • ALL stays     (≥1 ECG)  –   name="All stays"
    • MULTI stays   (≥2 ECGs) –   name="MULTI stays"
…then plots AUROC and AUPRC curves comparing the two regimes.
All metrics, confusion matrices, ROC/AUPRC data, and loss curves
are logged in folders structured as:
    outputs/<arch>/{all_performance|multi_performance}/
Plus combined plots in    outputs/<arch>/combined_performance/

Usage
-----
$ python train_ecg_only_models.py gru
$ python train_ecg_only_models.py transformer
$ python train_ecg_only_models.py tcn
(or omit arg to train all three sequentially)

Suggested tweaks
----------------
• For Transformer increase ECG_EMB_DIM to 512 (done below) and set nhead so
  that ECG_EMB_DIM % nhead == 0.
• TCN receptive‑field scales with N_LAYERS & DILATIONS; adjust if ECG count » 6.
• Consider a cosine LR scheduler if you extend training >20 epochs.
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

import wfdb, neurokit2 as nk
from sklearn.experimental import enable_iterative_imputer  # noqa: F401 – keeps old import path working
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve, confusion_matrix, precision_recall_fscore_support

# ─────────────────── Global hyper‑parameters ───────────────────
CSV_ECG          = "final_ecgs.csv"
WAVE_DIR         = "..\\ecg"
ENC_CKPT         = "pretrained_model.pth"

DEVICE           = "cuda" if torch.cuda.is_available() else "cpu"

MAX_ECG_SEQ      = 6      # keep last N ECGs
ECG_SEQ_LEN      = 5_000  # samples per lead
ECG_EMB_DIM      = 512  # output of ResNet18+adapter
SEQ_FEAT_DIM  = ECG_EMB_DIM + 1 
HIDDEN           = 256    # GRU hidden & Transformer model dim
BATCH_SIZE       = 32
EPOCHS           = 20
POS_WEIGHT       = 1.0
PATIENCE_ES = 3
LR_NET           = 3e-4
LR_ADAPTER       = 1e-4
LR_FINETUNE      = 5e-5
LR_GAMMA         = 0.9    # exponential decay factor each epoch
base_lr = 1e-4  
SEED             = 10
record_df = pd.read_csv("record_list.csv", parse_dates=["ecg_time"])
PATH2TIME = dict(zip(record_df["path"], record_df["ecg_time"]))

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# ─────────────────────── Utility funcs ────────────────────────
def set_seed(seed:int=SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

pd.DataFrame.pad = pd.DataFrame.ffill  # legacy compat
pd.Series.pad    = pd.Series.ffill

warnings.filterwarnings(
    "ignore",
    message=r"There are .* missing data points in your signal.*",
    category=UserWarning,
    module=r".*neurokit2\.ecg\.ecg_clean.*",
)

# ─────────────────────── ECG loader ───────────────────────────

def load_ecg(path:str) -> torch.Tensor:
    """Return a (12, ECG_SEQ_LEN) float32 tensor."""
    rec = wfdb.rdrecord(path)
    sig = rec.p_signal.astype(float)
    sig[~np.isfinite(sig)] = 0.0
    flat = (np.abs(sig).sum(axis=0) == 0)
    if flat.any():
        sig[:, flat] += 1e-6 * np.random.randn(sig.shape[0], flat.sum())
    cleaned = np.stack([
        nk.ecg_clean(sig[:, j], sampling_rate=500, method="nk") for j in range(12)
    ], axis=0)
    if cleaned.shape[1] >= ECG_SEQ_LEN:
        cleaned = cleaned[:, :ECG_SEQ_LEN]
    else:
        cleaned = np.pad(cleaned, ((0,0), (0, ECG_SEQ_LEN - cleaned.shape[1])))
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

def resnet18_backbone():
    return ResNet1D(BasicBlock1D, [2,2,2,2])

class ECGEncoder(nn.Module):
    def __init__(self, ckpt=ENC_CKPT, emb_dim=ECG_EMB_DIM):
        super().__init__()
        self.backbone = resnet18_backbone()
        sd = torch.load(ckpt, map_location="cpu")
        self.backbone.load_state_dict({k:v for k,v in sd.items() if k in self.backbone.state_dict() and "fc" not in k}, strict=False)
        self.adapter = nn.Linear(512, emb_dim)
        torch.nn.init.kaiming_normal_(self.adapter.weight, nonlinearity='relu')
    def forward(self, x):
        return self.adapter(self.backbone(x))




# ─────────────────────── Dataset ─────────────────────────────
ecg_df = pd.read_csv(CSV_ECG, parse_dates=["ecg_time"])

class StayECG(Dataset):
    """Return (ecg_seq, label)"""
    def __init__(self, stay_ids):
        self.stay_ids = stay_ids
        self.ecg_df   = ecg_df.set_index("stay_id")
        self.bad      = set()
    def __len__(self): return len(self.stay_ids)
    def __getitem__(self, idx):
        sid  = self.stay_ids[idx]
        row  = self.ecg_df.loc[sid]

        # final ECG timestamp is already in ecg_time
        t_final = row["ecg_time"]


        paths = [row.get(f"path_{i}") for i in range(1, MAX_ECG_SEQ)]
        paths = [p for p in paths if isinstance(p, str) and p] + [row["final_ecg_path"]]
        paths = paths[-MAX_ECG_SEQ:]
        sig_list, dt_list = [], []

        for p in paths:
            full = os.path.join(WAVE_DIR, p)
            try:
                sig = load_ecg(full)          # (12, 5000)
                sig_list.append(sig)

                # time delta in hours (clip at 0; earlier ECGs only)
                t = PATH2TIME.get(p, pd.NaT)
                dt_hours = max((t_final - t).total_seconds() / 3600.0, 0.0)
                dt_list.append(dt_hours)

            except Exception as e:
                warnings.warn(f"...")

        if not sig_list:
            raise ValueError(f"No usable ECGs for {sid}")
        ecg_seq = torch.stack(sig_list)                 # (L, 12, 5000)
        dt_seq  = torch.tensor(dt_list, dtype=torch.float32)  # (L,)
        label   = torch.tensor(float(row["disposition_HOME"]), dtype=torch.float32)
        return ecg_seq, dt_seq, label



# ─────────────────── collate (variable length) ───────────────

def collate(batch):
    ecg, dt, y = zip(*batch)
    lens = torch.tensor([t.size(0) for t in ecg])
    y    = torch.stack(y)

    ecg = pad_sequence(ecg, batch_first=True)       # (B, T, 12, 5000)
    dt  = pad_sequence(dt,  batch_first=True)       # (B, T)
    return ecg, dt, lens, y


# ─────────────────── Model definitions ───────────────────────
class GRUHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru  = nn.GRU(SEQ_FEAT_DIM, HIDDEN, batch_first=True)
        self.drop = nn.Dropout(0.3)
        self.out  = nn.Linear(HIDDEN, 1)
    def _last(self, x, lens):
        packed = pack_padded_sequence(x, lens.cpu(), batch_first=True, enforce_sorted=False)
        _, h = self.gru(packed)
        return h.squeeze(0)
    def forward(self, x, lens):
        h = self._last(x, lens)
        return self.out(self.drop(h)).squeeze(1)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=MAX_ECG_SEQ):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10_000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))           # (1,max_len,d_model)
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class TransformerHead(nn.Module):
    def __init__(self, nhead=8, nlayers=2):
        super().__init__()
        self.pos = PositionalEncoding(SEQ_FEAT_DIM)
        enc_layer = nn.TransformerEncoderLayer(SEQ_FEAT_DIM, nhead, dim_feedforward=SEQ_FEAT_DIM*4, dropout=0.1, batch_first=True)
        self.enc = nn.TransformerEncoder(enc_layer, nlayers)
        self.out = nn.Linear(SEQ_FEAT_DIM, 1)
    def forward(self, x, lens):
        mask = torch.arange(x.size(1), device=x.device)[None,:] >= lens[:,None]
        h = self.enc(self.pos(x), src_key_padding_mask=mask)
        h = h.gather(1, (lens-1).view(-1,1,1).expand(-1,1,h.size(2))).squeeze(1)  # last valid token
        return self.out(h).squeeze(1)

class Chomp1d(nn.Module):
    def __init__(self, chomp):
        super().__init__(); self.chomp = chomp
    def forward(self, x):
        return x[:, :, :-self.chomp]
class TemporalBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k, d, drop):
        super().__init__()
        pad = (k-1)*d
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, k, padding=pad, dilation=d),
            Chomp1d(pad), nn.ReLU(), nn.Dropout(drop),
            nn.Conv1d(out_ch, out_ch, k, padding=pad, dilation=d),
            Chomp1d(pad), nn.ReLU(), nn.Dropout(drop))
        self.down = nn.Conv1d(in_ch, out_ch, 1) if in_ch!=out_ch else nn.Identity()
    def forward(self, x):
        return torch.relu(self.net(x) + self.down(x))
class TCNHead(nn.Module):
    def __init__(self, levels=(HIDDEN, HIDDEN)):
        super().__init__()
        layers = []
        in_ch = SEQ_FEAT_DIM
        for i,out_ch in enumerate(levels):
            layers.append(TemporalBlock(in_ch, out_ch, k=3, d=2**i, drop=0.1))
            in_ch = out_ch
        self.tcn = nn.Sequential(*layers)
        self.out = nn.Linear(in_ch, 1)
    def forward(self, x, lens):
        # x: (B,T,EMB) → (B,EMB,T)
        y = self.tcn(x.transpose(1,2))              # (B,C,T)
        idx = (lens-1).view(-1,1,1).expand(-1, y.size(1), 1)
        last = y.gather(2, idx).squeeze(2)          # (B,C)
        return self.out(last).squeeze(1)

HEADS = {
    "gru": GRUHead,
    "transformer": TransformerHead,
    "tcn": TCNHead,
}

# ─────────────── Train / eval helpers ────────────────────────
BCE = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(POS_WEIGHT, device=DEVICE))

def epoch_pass(head, loader, opt=None, desc="train"):
    train = opt is not None
    head.train() if train else head.eval()
    encoder.train() if train else encoder.eval()    # switch BN mode

    yt, yp, losses = [], [], []
    bar = tqdm(loader, desc=desc, leave=False)

    for ecg, dt, lens, y in bar: 
        # move to GPU *once*
        ecg  = ecg.to(DEVICE)        # (B,T,12,5000)
        y    = y.to(DEVICE)

        B, T, C, L = ecg.shape
        ecg_flat = ecg.view(B*T, C, L)              # (B*T,12,5000)
        emb_flat = encoder(ecg_flat)                # (B*T,512)
        emb = emb_flat.view(B, T, -1)  
        dt = dt.to(DEVICE).unsqueeze(-1)       # (B, T, 1)
        emb = torch.cat([emb, dt], dim=-1)              # (B,T,512)

        with torch.set_grad_enabled(train):
            logit = head(emb, lens.to(DEVICE))
            loss  = BCE(logit, y)
            if train:
                opt.zero_grad()
                loss.backward()
                opt.step()

        yt.append(y.detach().cpu())
        yp.append(torch.sigmoid(logit).detach().cpu())
        losses.append(loss.item())
        bar.set_postfix(loss=f"{loss.item():.3f}")

    yt = torch.cat(yt).numpy(); yp = torch.cat(yp).numpy()
    return np.mean(losses), roc_auc_score(yt, yp), average_precision_score(yt, yp), yt, yp

def roc_prc_arrays(y_true, y_prob):
    fpr, tpr, _ = roc_curve(y_true, y_prob); prec, rec, _ = precision_recall_curve(y_true, y_prob)
    return fpr, tpr, prec, rec

# ──────────────── train_variant (one regime) ────────────────

def train_variant(head_cls, name, keep_multi, perf_dir):
    global encoder
    encoder = ECGEncoder().to(DEVICE)
    encoder.eval()
    for p in encoder.backbone.parameters():          # ← everything
        p.requires_grad = True
    set_seed()
    stays = ecg_df.loc[ecg_df["n_ecg"].ge(2) if keep_multi else slice(None), "stay_id"].tolist()
    random.shuffle(stays); split = int(0.9*len(stays))
    tr_ids, va_ids = stays[:split], stays[split:]

    tr_ds, va_ds = StayECG(tr_ids), StayECG(va_ids)
    tr_dl = DataLoader(tr_ds, BATCH_SIZE, True,  collate_fn=collate)
    va_dl = DataLoader(va_ds, BATCH_SIZE, False, collate_fn=collate)

    head = head_cls().to(DEVICE)

    opt = torch.optim.AdamW([
        # head and adapter – highest lr (same as before)
        {"params": head.parameters(),                  "lr": 3e-4},
        {"params": encoder.adapter.parameters(),       "lr": 3e-4},

        # ResNet blocks with progressively smaller lrs
        {"params": encoder.backbone.layer4.parameters(), "lr": base_lr},
        {"params": encoder.backbone.layer3.parameters(), "lr": base_lr / 2},
        {"params": encoder.backbone.layer2.parameters(), "lr": base_lr / 4},
        {"params": encoder.backbone.layer1.parameters(), "lr": base_lr / 8},
    ], weight_decay=1e-4)

    sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=LR_GAMMA)

    stem = name.lower().replace(" ", "_")
    perf_dir.mkdir(exist_ok=True)
    history, best_auc = [], -np.inf
    best, no_improve  = dict(auc=-np.inf), 0        
    for ep in trange(1, EPOCHS+1, desc=f"{name} epochs"):
        tr_loss, tr_auc, _, _, _ = epoch_pass(head, tr_dl, opt, "train")
        va_loss, va_auc, va_ap, yt, yp = epoch_pass(head, va_dl, None, "valid")
        improved = va_auc > best_auc + 1e-4 
        if improved:
            best_auc, no_improve = va_auc, 0
            best.update(
                epoch = ep,
                auc   = va_auc,
                ap    = va_ap,
                yt    = yt.copy(),        # store predictions
                yp    = yp.copy()
            )
            torch.save(
                {"epoch": ep,
                "val_auc": va_auc,
                "model": head.state_dict(),
                "encoder": encoder.state_dict()},
                perf_dir / f"{stem}_best.pth"
            )        
        else:
            no_improve += 1
        tqdm.write(f"[{name}] ep{ep:02d} LR{sched.get_last_lr()[0]:.1e} │ tr_loss{tr_loss:.3f} AUC{tr_auc:.3f} │ val_loss{va_loss:.3f} AUC{va_auc:.3f}")
        history.append({"epoch":ep,"train_loss":tr_loss,"val_loss":va_loss,"train_auc":tr_auc,"val_auc":va_auc,"val_ap":va_ap})
        if no_improve >= PATIENCE_ES:
            tqdm.write(f"↳ Early stopping after epoch {ep} "
                   f"(no val-AUC improvement for {PATIENCE_ES} epochs)")
            break
        sched.step()
    yt, yp      = best["yt"], best["yp"]   # ← BEST epoch, not last
    va_auc      = best["auc"]
    va_ap       = best["ap"]
    best_epoch  = best["epoch"]
    pred = (yp>0.5).astype(int)
    tn,fp,fn,tp = confusion_matrix(yt, pred).ravel()
    pr,rc,f1,_  = precision_recall_fscore_support(yt,pred,average="binary",zero_division=0)

    df = pd.DataFrame(history)
    df["is_best"] = df["epoch"] == best_epoch
    df.to_csv(perf_dir / f"{stem}_metrics.csv", index=False)

    pd.DataFrame([{
        "epoch"    : best_epoch,
        "tn"       : tn, "fp": fp, "fn": fn, "tp": tp,
        "precision": pr, "recall": rc, "f1": f1,
        "val_auc"  : va_auc,
        "val_ap"   : va_ap
    }]).to_csv(perf_dir / f"{stem}_confusion.csv", index=False)
    # loss plot
    plt.figure(figsize=(5,3))
    epochs=[h["epoch"] for h in history]
    plt.plot(epochs,[h["train_loss"] for h in history],label="train")
    plt.plot(epochs,[h["val_loss"] for h in history],label="val")
    plt.tight_layout(); plt.legend(); plt.xlabel("epoch"); plt.ylabel("BCE loss")
    plt.savefig(perf_dir/f"{stem}_loss.png",dpi=200); plt.close()

    fpr,tpr,prec,rec = roc_prc_arrays(yt, yp)
    np.savez(perf_dir/f"{stem}_roc_prc.npz",fpr=fpr,tpr=tpr,prec=prec,rec=rec,auc=va_auc,ap=va_ap)
    return {"fpr":fpr,"tpr":tpr,"prec":prec,"rec":rec,"auc":va_auc,"ap":va_ap}

# ──────────────── combined plots per arch ───────────────────

def combined_plots(out_dir, multi, single, arch_tag):
    out_dir.mkdir(exist_ok=True)
    # ROC
    plt.figure();
    plt.plot(multi["fpr"], multi["tpr"],label=f"MULTI stays (AUC {multi['auc']:.3f})")
    plt.plot(single["fpr"],single["tpr"],label=f"All stays (AUC {single['auc']:.3f})")
    plt.plot([0,1],[0,1],'k--',alpha=.4); plt.xlabel("FPR"); plt.ylabel("TPR")
    plt.title(f"ROC – {arch_tag}"); plt.legend(); plt.tight_layout()
    plt.savefig(out_dir/"combined_roc.png",dpi=200); plt.close()
    # PRC
    plt.figure();
    plt.plot(multi["rec"], multi["prec"],label=f"MULTI stays (AUPRC {multi['ap']:.3f})")
    plt.plot(single["rec"],single["prec"],label=f"All stays (AUPRC {single['ap']:.3f})")
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title(f"PRC – {arch_tag}"); plt.legend(); plt.tight_layout()
    plt.savefig(out_dir/"combined_prc.png",dpi=200); plt.close()

# ───────────────────────── main ─────────────────────────────

def main(arch_list):
    for arch in arch_list:
        head_cls = HEADS[arch]
        base_out = Path("outputs")/arch
        # MULTI first (seed reset inside)
        multi = train_variant(head_cls, "MULTI stays", keep_multi=True,  perf_dir=base_out/"multi_performance")
        single= train_variant(head_cls, "All stays",   keep_multi=False, perf_dir=base_out/"all_performance")
        combined_plots(base_out/"combined_performance", multi, single, arch_tag=arch.upper())

if __name__ == "__main__":
    if len(sys.argv)==1:
        arches = ["gru"]
    else:
        arches = [a.lower() for a in sys.argv[1:]]
    for a in arches:
        assert a in HEADS, f"Unknown architecture '{a}'. Choose from {list(HEADS)}"
    main(arches)





