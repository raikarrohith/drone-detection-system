"""
Distance Residual GRU Model:
Predicts the systematic residual delta_D = D_ground_truth - D_D2 based on temporal sequences of
bounding box dynamics, perspective aspect ratios, CRLB uncertainty, and velocity.
Final Corrected Distance: D_corrected = D_D2 + delta_D
"""

import os
import json
import math
from pathlib import Path
from collections import deque
import numpy as np
import torch
import torch.nn as nn


class DroneDistanceGRU(nn.Module):
    """
    Lightweight 2-layer GRU for real-time distance residual regression.
    Input: Sequence of shape (Batch, Seq_Len=10, Input_Dim=9)
    Output: delta_D (Batch, 1) residual correction in meters.
    """
    def __init__(self, input_dim=9, hidden_dim=32, num_layers=2, dropout=0.10):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        # x: (B, T, input_dim)
        gru_out, _ = self.gru(x)
        # Take final time-step representation
        last_step = gru_out[:, -1, :]
        delta_d = self.head(last_step)
        return delta_d


class DistanceResidualPredictor:
    """
    Runtime helper that manages feature extraction, normalization, and GRU inference.
    """
    FEATURE_NAMES = [
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

    def __init__(self, model_path="models/distance_gru.pth", scaler_path="models/distance_gru_scaler.json", seq_len=10, device="cpu"):
        self.seq_len = seq_len
        self.device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
        self.model = None
        self.scaler = None
        self.enabled = False
        
        self.load(model_path, scaler_path)

    def load(self, model_path, scaler_path):
        m_path = Path(model_path)
        s_path = Path(scaler_path)
        
        if not m_path.is_file() or not s_path.is_file():
            self.enabled = False
            return False
            
        try:
            with open(s_path, "r") as f:
                self.scaler = json.load(f)
                
            self.mean = np.array(self.scaler["mean"], dtype=np.float32)
            self.std = np.array(self.scaler["std"], dtype=np.float32)
            # Avoid division by zero
            self.std = np.where(self.std < 1e-6, 1.0, self.std)
            
            input_dim = len(self.mean)
            hidden_dim = self.scaler.get("hidden_dim", 32)
            num_layers = self.scaler.get("num_layers", 2)
            
            self.model = DroneDistanceGRU(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers)
            state_dict = torch.load(m_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            self.model.eval()
            self.enabled = True
            return True
        except Exception as e:
            print(f"[!] Warning loading Distance Residual GRU: {e}")
            self.enabled = False
            return False

    def normalize(self, feat_array):
        return (feat_array - self.mean) / self.std

    def predict_correction(self, feature_sequence):
        """
        feature_sequence: list or array of shape (N, 9)
        Returns: delta_D in meters (float)
        """
        if not self.enabled or self.model is None or len(feature_sequence) == 0:
            return 0.0
            
        try:
            if len(feature_sequence) < self.seq_len:
                # Warmup padding: repeat the earliest available frame to fill 10-step buffer
                first_elem = feature_sequence[0]
                pad_count = self.seq_len - len(feature_sequence)
                padded_seq = [first_elem] * pad_count + list(feature_sequence)
                seq_np = np.array(padded_seq, dtype=np.float32)
            else:
                seq_np = np.array(feature_sequence[-self.seq_len:], dtype=np.float32)

            norm_seq = self.normalize(seq_np)
            tensor = torch.tensor(norm_seq, dtype=torch.float32).unsqueeze(0).to(self.device)
            with torch.no_grad():
                delta_d = float(self.model(tensor).squeeze().item())
            # Safety clamp: do not allow wild residual shifts (> 3.0m)
            delta_d = max(-3.0, min(3.0, delta_d))
            return delta_d
        except Exception:
            return 0.0
