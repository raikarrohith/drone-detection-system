#!/usr/bin/env python3
"""
Train Distance Residual GRU Model:
Trains a lightweight 2-layer GRU network to predict the systematic residual delta_D = GT_distance - D2_distance.
Key Implementation Principles:
- Split dataset strictly by FLIGHT / VIDEO SEQUENCE (70% train, 15% val, 15% test) to prevent temporal leakage.
- Normalizes features using mean/std calculated ONLY on the training split.
- Huber Loss / SmoothL1Loss for robust regression against visual jitter outliers.
- Early stopping, learning rate scheduling, model checkpointing.
- Saves model weights to models/distance_gru.pth and scaler parameters to models/distance_gru_scaler.json.
"""

import os
import sys
import json
import math
import random
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from distance_residual_gru import DroneDistanceGRU

FEATURE_KEYS = [
    "d2_distance",
    "box_w",
    "box_h",
    "aspect_ratio",
    "delta_d2",
    "velocity_2d",
    "confidence",
    "crlb_sigma",
    "kalman_pred_z"
]


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class SequenceDataset(Dataset):
    def __init__(self, sequences, seq_len=10, mean=None, std=None):
        self.seq_len = seq_len
        self.samples = []
        self.targets = []
        
        # Build sliding windows of length seq_len within each sequence
        for seq in sequences:
            if len(seq) < seq_len:
                continue
            feats = np.array([[frame[k] for k in FEATURE_KEYS] for frame in seq], dtype=np.float32)
            resids = np.array([frame["residual"] for frame in seq], dtype=np.float32)
            
            for i in range(len(seq) - seq_len + 1):
                window_feats = feats[i : i + seq_len]
                target_resid = resids[i + seq_len - 1]
                self.samples.append(window_feats)
                self.targets.append(target_resid)
                
        self.samples = np.array(self.samples, dtype=np.float32)
        self.targets = np.array(self.targets, dtype=np.float32)
        
        # Normalize features
        if mean is not None and std is not None:
            std_safe = np.where(std < 1e-6, 1.0, std)
            self.samples = (self.samples - mean) / std_safe

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return torch.tensor(self.samples[idx], dtype=torch.float32), torch.tensor([self.targets[idx]], dtype=torch.float32)


def load_and_split_data(json_path, seq_len=10, train_ratio=0.70, val_ratio=0.15, seed=42):
    with open(json_path, "r") as f:
        data = json.load(f)
        
    # Group frames by sequence_id
    seq_dict = {}
    for item in data:
        s_id = item["sequence_id"]
        if s_id not in seq_dict:
            seq_dict[s_id] = []
        seq_dict[s_id].append(item)
        
    # Sort frames within each sequence by frame_idx
    for s_id in seq_dict:
        seq_dict[s_id].sort(key=lambda x: x["frame_idx"])
        
    sequence_ids = sorted(list(seq_dict.keys()))
    
    # Shuffle sequence IDs reproducibly
    rng = random.Random(seed)
    rng.shuffle(sequence_ids)
    
    n_total = len(sequence_ids)
    n_train = max(1, int(round(n_total * train_ratio)))
    n_val = max(1, int(round(n_total * val_ratio)))
    
    train_ids = sequence_ids[:n_train]
    val_ids = sequence_ids[n_train : n_train + n_val]
    test_ids = sequence_ids[n_train + n_val:]
    
    # Ensure test has at least one sequence
    if not test_ids and len(val_ids) > 1:
        test_ids = [val_ids.pop()]
        
    train_seqs = [seq_dict[sid] for sid in train_ids]
    val_seqs = [seq_dict[sid] for sid in val_ids]
    test_seqs = [seq_dict[sid] for sid in test_ids]
    
    print(f"[*] Dataset Split Summary:")
    print(f"  - Total Sequences: {n_total}")
    print(f"  - Train: {len(train_ids)} seqs ({sum(len(s) for s in train_seqs)} frames) -> {train_ids[:3]}...")
    print(f"  - Val:   {len(val_ids)} seqs ({sum(len(s) for s in val_seqs)} frames) -> {val_ids}")
    print(f"  - Test:  {len(test_ids)} seqs ({sum(len(s) for s in test_seqs)} frames) -> {test_ids}")
    
    # Compute feature scaler ONLY on training sequences
    all_train_feats = []
    for seq in train_seqs:
        for frame in seq:
            all_train_feats.append([frame[k] for k in FEATURE_KEYS])
    all_train_feats = np.array(all_train_feats, dtype=np.float32)
    
    mean = np.mean(all_train_feats, axis=0)
    std = np.std(all_train_feats, axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    
    return train_seqs, val_seqs, test_seqs, mean, std, test_ids


def train_model(
    json_path="datasets/distance_sequences.json",
    models_dir="models",
    epochs=100,
    batch_size=32,
    lr=0.003,
    hidden_dim=32,
    num_layers=2,
    dropout=0.10,
    seq_len=10,
    seed=42
):
    set_seed(seed)
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    
    train_seqs, val_seqs, test_seqs, mean, std, test_ids = load_and_split_data(
        json_path, seq_len=seq_len, seed=seed
    )
    
    train_ds = SequenceDataset(train_seqs, seq_len=seq_len, mean=mean, std=std)
    val_ds = SequenceDataset(val_seqs, seq_len=seq_len, mean=mean, std=std)
    test_ds = SequenceDataset(test_seqs, seq_len=seq_len, mean=mean, std=std)
    
    print(f"[*] Sliding Window Samples (N={seq_len}):")
    print(f"  - Train Samples: {len(train_ds)}")
    print(f"  - Val Samples:   {len(val_ds)}")
    print(f"  - Test Samples:  {len(test_ds)}")
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Training on compute device: {device}")
    
    model = DroneDistanceGRU(
        input_dim=len(FEATURE_KEYS),
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout
    ).to(device)
    
    criterion = nn.SmoothL1Loss(beta=0.05)  # Huber Loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=6)
    
    best_val_loss = float("inf")
    patience_counter = 0
    patience_limit = 18
    best_state_dict = None
    
    print("\n" + "=" * 70)
    print(f"  TRAINING DISTANCE RESIDUAL GRU (Epochs: {epochs}, Dim: {hidden_dim}, Layers: {num_layers})")
    print("=" * 70)
    
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for x_b, y_b in train_loader:
            x_b, y_b = x_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            pred = model(x_b)
            loss = criterion(pred, y_b)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            train_loss += loss.item() * len(x_b)
            
        train_loss /= max(1, len(train_ds))
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_mae = 0.0
        with torch.no_grad():
            for x_v, y_v in val_loader:
                x_v, y_v = x_v.to(device), y_v.to(device)
                pred_v = model(x_v)
                v_loss = criterion(pred_v, y_v)
                val_loss += v_loss.item() * len(x_v)
                val_mae += torch.sum(torch.abs(pred_v - y_v)).item()
                
        val_loss /= max(1, len(val_ds))
        val_mae /= max(1, len(val_ds))
        scheduler.step(val_loss)
        
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            best_state_dict = model.state_dict().copy()
            patience_counter = 0
            tag = " [BEST SAVED]"
        else:
            patience_counter += 1
            tag = ""
            
        if epoch % 5 == 0 or is_best or epoch == 1:
            print(f"  Epoch {epoch:03d}/{epochs:03d} | Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f} | Val Residual MAE: {val_mae*100:.2f}cm{tag}")
            
        if patience_counter >= patience_limit:
            print(f"[*] Early stopping triggered at epoch {epoch} (no val loss improvement for {patience_limit} epochs).")
            break
            
    # Save best model checkpoint
    model_save_path = models_dir / "distance_gru.pth"
    torch.save(best_state_dict, model_save_path)
    print(f"[+] Saved model checkpoint to: {model_save_path}")
    
    # Save feature scaler & architecture parameters
    scaler_dict = {
        "feature_names": FEATURE_KEYS,
        "mean": mean.tolist(),
        "std": std.tolist(),
        "seq_len": seq_len,
        "input_dim": len(FEATURE_KEYS),
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
        "test_sequence_ids": test_ids
    }
    scaler_save_path = models_dir / "distance_gru_scaler.json"
    with open(scaler_save_path, "w") as f:
        json.dump(scaler_dict, f, indent=2)
    print(f"[+] Saved scaler configuration to: {scaler_save_path}")
    
    # Evaluate on Unseen Test Sequences
    model.load_state_dict(best_state_dict)
    model.eval()
    test_loss = 0.0
    test_mae = 0.0
    with torch.no_grad():
        for x_t, y_t in test_loader:
            x_t, y_t = x_t.to(device), y_t.to(device)
            p_t = model(x_t)
            t_loss = criterion(p_t, y_t)
            test_loss += t_loss.item() * len(x_t)
            test_mae += torch.sum(torch.abs(p_t - y_t)).item()
    test_loss /= max(1, len(test_ds))
    test_mae /= max(1, len(test_ds))
    
    print("=" * 70)
    print(f"[+] Training complete! Test Loss: {test_loss:.5f} | Test Residual MAE: {test_mae*100:.2f} cm")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Distance Residual GRU")
    parser.add_argument("--data", type=str, default="datasets/distance_sequences.json", help="Path to sequence dataset JSON")
    parser.add_argument("--epochs", type=int, default=80, help="Max training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.003, help="Initial learning rate")
    parser.add_argument("--hidden-dim", type=int, default=32, help="GRU hidden dimension")
    parser.add_argument("--num-layers", type=int, default=2, help="GRU number of layers")
    parser.add_argument("--seq-len", type=int, default=10, help="Sliding sequence length N")
    args = parser.parse_args()

    train_model(
        json_path=args.data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        seq_len=args.seq_len
    )
