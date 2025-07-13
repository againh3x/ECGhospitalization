import argparse
import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import wfdb
import neurokit2 as nk
from tqdm import tqdm

class ECGDataset(Dataset):
    def __init__(self, csv_file, ecg_root, num_cols=None, desired_length=5000):
        self.df = pd.read_csv(csv_file)
        self.ecg_root = ecg_root
        if num_cols is None:
            num_cols = ["rr_interval", "qrs_onset", "qrs_end",
                        "t_end", "qrs_axis", "t_axis", "qrs_duration"]
        self.num_cols = num_cols
        self.bool_cols = [c for c in self.df.columns
                          if c not in ["path"] + self.num_cols]
        self.desired_length = desired_length

        # 1) compute and store normalization stats
        means = self.df[self.num_cols].mean()
        stds  = self.df[self.num_cols].std().replace(0, 1e-6)
        self.num_means = means
        self.num_stds  = stds

        # 2) apply z-score normalization
        self.df.loc[:, self.num_cols] = (self.df[self.num_cols] - means) / stds

        self.num_cols = num_cols
        

    def __len__(self):  
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        record_rel = row['path']
        record_path = os.path.join(self.ecg_root, record_rel)
        try:
            record = wfdb.rdrecord(record_path)
            ecg_signal = record.p_signal  # shape: (N_samples, 12)
            # Clean each lead with robust error handling
            cleaned = []
            for i in range(ecg_signal.shape[1]):
                lead = ecg_signal[:, i]
                # if entire lead is zero, skip record
                if np.all(lead == 0):
                    return None
                try:
                    clean_lead = nk.ecg_clean(lead, sampling_rate=500)
                except Exception:
                    return None
                cleaned.append(clean_lead)
            cleaned = np.stack(cleaned, axis=0)  # shape: (12, N_samples)
            # pad/truncate to fixed length
            _, orig_len = cleaned.shape
            if orig_len > self.desired_length:
                cleaned = cleaned[:, :self.desired_length]
            elif orig_len < self.desired_length:
                pad_width = self.desired_length - orig_len
                cleaned = np.pad(cleaned, ((0,0),(0,pad_width)), 'constant')
            signal = torch.tensor(cleaned, dtype=torch.float)
            bool_targets = torch.tensor(row[self.bool_cols].values.astype(np.float32))
            num_targets = torch.tensor(row[self.num_cols].values.astype(np.float32))
            return signal, bool_targets, num_targets
        except Exception:
            return None

# Custom collate_fn to filter out None examples

def collate_fn(batch):
    batch = [x for x in batch if x is not None]
    if len(batch) == 0:
        return torch.empty(0), torch.empty(0), torch.empty(0)
    signals, bools, nums = zip(*batch)
    return torch.stack(signals), torch.stack(bools), torch.stack(nums)

# Basic 1D ResNet building blocks
class BasicBlock1D(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = None
        if stride != 1 or in_planes != planes:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes)
            )
    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            identity = self.downsample(identity)
        out += identity
        return self.relu(out)

class ResNet1D(nn.Module):
    def __init__(self, block, layers, num_bool, num_reg):
        super().__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv1d(12, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc_bool = nn.Linear(512 * block.expansion, num_bool)
        self.fc_reg = nn.Linear(512 * block.expansion, num_reg)
    def _make_layer(self, block, planes, blocks, stride=1):
        strides = [stride] + [1]*(blocks-1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x).squeeze(-1)
        return self.fc_bool(x), self.fc_reg(x)

def resnet18_1d(num_bool, num_reg):
    return ResNet1D(BasicBlock1D, [2,2,2,2], num_bool, num_reg)

# Training loop

def train(args):
    dataset = ECGDataset(args.csv, args.ecg_dir, num_cols=args.num_cols, desired_length=args.seq_len)
    loader = DataLoader(dataset,
                        batch_size=args.batch_size,
                        shuffle=True,
                        num_workers=args.num_workers,
                        pin_memory=True,
                        collate_fn=collate_fn)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = resnet18_1d(len(dataset.bool_cols), len(dataset.num_cols)).to(device)
    criterion_bool = nn.BCEWithLogitsLoss()
    criterion_reg  = nn.MSELoss()
    optimizer      = optim.Adam(model.parameters(), lr=args.lr)

    epoch_stats = []
    for epoch in range(1, args.epochs+1):
        model.train()
        sum_bool = 0.0
        sum_reg  = 0.0
        total    = 0
        for signals, bool_t, num_t in tqdm(loader, desc=f"Epoch {epoch}"):
            signals, bool_t, num_t = signals.to(device), bool_t.to(device), num_t.to(device)
            optimizer.zero_grad()
            p_bool, p_reg = model(signals)
            loss_bool = criterion_bool(p_bool, bool_t)
            loss_reg  = criterion_reg(p_reg, num_t)
            loss      = 0.9 * loss_bool + 0.1 * loss_reg
            loss.backward()
            optimizer.step()

            batch_size = signals.size(0)
            sum_bool  += loss_bool.item() * batch_size
            sum_reg   += loss_reg.item()  * batch_size
            total     += batch_size

        avg_bool = sum_bool / total
        avg_reg  = sum_reg  / total
        avg_tot  = 0.9 * avg_bool + 0.1 * avg_reg

        print(f"Epoch {epoch}/{args.epochs}  │  "
              f"BoolLoss: {avg_bool:.4f}  │  "
              f"RegLoss: {avg_reg:.4f}  │  "
              f"Total: {avg_tot:.4f}")

        epoch_stats.append({
            "epoch": epoch,
            "bool_loss": avg_bool,
            "reg_loss": avg_reg,
            "total_loss": avg_tot
        })

    # Save model
    torch.save(model.state_dict(), args.output)
    print(f"Model saved to {args.output}")

    # Dump tracking CSVN
    pd.DataFrame(epoch_stats).to_csv("loss_history.csv", index=False)
    print("Wrote loss_history.csv")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv',       required=True)
    parser.add_argument('--ecg_dir',   required=True)
    parser.add_argument('--batch_size',type=int, default=32)
    parser.add_argument('--epochs',    type=int, default=10)
    parser.add_argument('--lr',        type=float, default=1e-3)
    parser.add_argument('--num_workers',type=int,default=4)
    parser.add_argument('--seq_len',   type=int, default=5000)
    parser.add_argument('--output',    default='pretrained_model.pth')
    parser.add_argument('--num_cols',  nargs='+',
                        default=["rr_interval","qrs_onset","qrs_end",
                                 "t_end","qrs_axis","t_axis", "qrs_duration"])
    args = parser.parse_args()
    train(args)
