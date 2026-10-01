#!/usr/bin/env python3
"""
Comprehensive Distance Estimation Evaluation Engine:
Evaluates and benchmarks all 4 distance estimation pipelines on unseen test sequences:
  Method A: D2 only (Baseline raw optical geometry)
  Method B: D2 + Kalman (Current system baseline ~7.28% error)
  Method C: D2 + GRU (Neural residual corrected optical geometry)
  Method D: D2 + GRU + Kalman (Proposed full hybrid pipeline)

Calculates:
- Mean Absolute Error (MAE in cm & m)
- Root Mean Squared Error (RMSE in cm & m)
- Mean Absolute Percentage Error (MAPE / %)
- Standard Deviation of Error (sigma_error)
- Maximum Absolute Error (Max Error in cm & %)
- Inference Latency (ms) and throughput (FPS)

Logs step-by-step telemetry to:
  flight_logs/gru_distance_evaluation.csv
"""

import os
import sys
import csv
import json
import time
import math
import argparse
from pathlib import Path
import numpy as np
import torch

from distance_residual_gru import DistanceResidualPredictor, DroneDistanceGRU

class Kalman1DEvaluator:
    def __init__(self, init_z, init_sigma=0.5):
        self.z = float(init_z)
        self.vz = 0.0
        self.p00 = max(0.04, float(init_sigma)**2)
        self.p01 = 0.0
        self.p10 = 0.0
        self.p11 = 4.0
        self.q_pos = 0.05
        self.q_vel = 0.35

    def step(self, dt, z_meas, crlb_variance):
        # Predict
        if dt <= 0 or dt > 1.5:
            dt = 0.033
        self.z += self.vz * dt
        new_p00 = self.p00 + dt * (self.p10 + self.p01) + (dt**2) * self.p11 + self.q_pos * dt
        new_p01 = self.p01 + dt * self.p11
        new_p10 = self.p10 + dt * self.p11
        new_p11 = self.p11 + self.q_vel * dt
        self.p00, self.p01, self.p10, self.p11 = new_p00, new_p01, new_p10, new_p11

        # Update
        r = max(1e-4, float(crlb_variance))
        y = z_meas - self.z
        s = self.p00 + r
        k0 = self.p00 / s
        k1 = self.p10 / s
        self.z += k0 * y
        self.vz += k1 * y
        self.p00 = max(1e-4, (1.0 - k0) * self.p00)
        self.p01 = (1.0 - k0) * self.p01
        self.p10 = -k1 * self.p00 + self.p10
        self.p11 = max(1e-3, -k1 * self.p01 + self.p11)
        return self.z, self.vz


def evaluate_pipelines(
    dataset_path="datasets/distance_sequences.json",
    model_path="models/distance_gru.pth",
    scaler_path="models/distance_gru_scaler.json",
    log_csv_path="flight_logs/gru_distance_evaluation.csv"
):
    with open(dataset_path, "r") as f:
        all_data = json.load(f)

    with open(scaler_path, "r") as f:
        scaler = json.load(f)

    test_seq_ids = scaler.get("test_sequence_ids", [])
    seq_len = scaler.get("seq_len", 10)
    feature_names = scaler.get("feature_names", [
        "d2_distance", "box_w", "box_h", "aspect_ratio", "delta_d2", "velocity_2d", "confidence", "crlb_sigma", "kalman_pred_z"
    ])

    predictor = DistanceResidualPredictor(model_path=model_path, scaler_path=scaler_path, seq_len=seq_len)
    if not predictor.enabled:
        print("[!] Error: Could not initialize DistanceResidualPredictor.")
        return

    # Group dataset into sequences
    seq_dict = {}
    for item in all_data:
        sid = item["sequence_id"]
        if sid not in seq_dict:
            seq_dict[sid] = []
        seq_dict[sid].append(item)

    for sid in seq_dict:
        seq_dict[sid].sort(key=lambda x: x["frame_idx"])

    # If test_seq_ids is specified, use only unseen test sequences
    if test_seq_ids:
        test_sequences = [seq_dict[sid] for sid in test_seq_ids if sid in seq_dict]
    else:
        # Fallback: take last 15% of sequences
        sorted_ids = sorted(list(seq_dict.keys()))
        n_test = max(1, int(len(sorted_ids) * 0.15))
        test_sequences = [seq_dict[sid] for sid in sorted_ids[-n_test:]]

    print("=" * 80)
    print("           DRONE DISTANCE ESTIMATION: 4-PIPELINE COMPARATIVE BENCHMARK")
    print("=" * 80)
    print(f"[*] Total Test Sequences: {len(test_sequences)} (Unseen during training)")
    print(f"[*] Sliding Feature Window: N = {seq_len} frames")
    print(f"[*] Evaluation Log: {log_csv_path}")

    # Metrics storage
    gt_all = []
    d2_only_all = []
    d2_kalman_all = []
    d2_gru_all = []
    d2_gru_kalman_all = []
    gru_latencies_ms = []

    # CSV log entries
    log_rows = []
    Path(log_csv_path).parent.mkdir(parents=True, exist_ok=True)

    for seq in test_sequences:
        seq_id = seq[0]["sequence_id"]
        # Initialize two independent Kalman filters: one for raw D2, one for GRU-corrected D2
        first_d2 = seq[0]["d2_distance"]
        kf_d2 = Kalman1DEvaluator(init_z=first_d2, init_sigma=seq[0]["crlb_sigma"])
        kf_gru = Kalman1DEvaluator(init_z=first_d2, init_sigma=seq[0]["crlb_sigma"])

        feature_buffer = []

        for frame_item in seq:
            gt_d = float(frame_item["ground_truth_distance"])
            d2_d = float(frame_item["d2_distance"])
            crlb_sig = float(frame_item["crlb_sigma"])
            crlb_var = crlb_sig ** 2
            ts = float(frame_item.get("timestamp", 0.0))
            dt = 0.0333

            # Method A: D2 only
            val_d2_only = d2_d

            # Method B: D2 + Kalman
            val_d2_kalman, _ = kf_d2.step(dt, d2_d, crlb_var)

            # Feature vector extraction for GRU
            feat_vec = [frame_item[k] for k in feature_names]
            feature_buffer.append(feat_vec)

            # GRU Residual Correction
            t_start = time.perf_counter()
            if len(feature_buffer) >= seq_len:
                delta_d = predictor.predict_correction(feature_buffer[-seq_len:])
            else:
                delta_d = 0.0  # Startup fallback
            lat_ms = (time.perf_counter() - t_start) * 1000.0
            gru_latencies_ms.append(lat_ms)

            # Method C: D2 + GRU
            val_d2_gru = d2_d + delta_d

            # Method D: D2 + GRU + Kalman
            val_d2_gru_kalman, _ = kf_gru.step(dt, val_d2_gru, crlb_var)

            # Record metrics
            gt_all.append(gt_d)
            d2_only_all.append(val_d2_only)
            d2_kalman_all.append(val_d2_kalman)
            d2_gru_all.append(val_d2_gru)
            d2_gru_kalman_all.append(val_d2_gru_kalman)

            # Write row to CSV
            log_rows.append({
                "timestamp": f"{ts:.4f}",
                "track_id": seq_id,
                "D2_distance": f"{d2_d:.4f}",
                "GRU_correction": f"{delta_d:.4f}",
                "corrected_distance": f"{val_d2_gru:.4f}",
                "CRLB_sigma": f"{crlb_sig:.4f}",
                "Kalman_distance": f"{val_d2_gru_kalman:.4f}",
                "ground_truth_distance": f"{gt_d:.4f}",
                "absolute_error": f"{abs(val_d2_gru_kalman - gt_d):.4f}"
            })

    # Save CSV
    with open(log_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "timestamp", "track_id", "D2_distance", "GRU_correction",
            "corrected_distance", "CRLB_sigma", "Kalman_distance",
            "ground_truth_distance", "absolute_error"
        ])
        writer.writeheader()
        writer.writerows(log_rows)

    # Convert to NumPy for metric computation
    gt = np.array(gt_all)
    m_a = np.array(d2_only_all)
    m_b = np.array(d2_kalman_all)
    m_c = np.array(d2_gru_all)
    m_d = np.array(d2_gru_kalman_all)

    def calc_stats(preds, targets):
        err = preds - targets
        abs_err = np.abs(err)
        mae = np.mean(abs_err)
        rmse = np.sqrt(np.mean(err**2))
        mape = np.mean(abs_err / np.maximum(1e-4, targets)) * 100.0
        std_err = np.std(err)
        max_err = np.max(abs_err)
        max_err_pct = np.max(abs_err / np.maximum(1e-4, targets)) * 100.0
        return {
            "mae_m": mae,
            "mae_cm": mae * 100.0,
            "rmse_m": rmse,
            "rmse_cm": rmse * 100.0,
            "mape_pct": mape,
            "std_cm": std_err * 100.0,
            "max_err_cm": max_err * 100.0,
            "max_err_pct": max_err_pct
        }

    stats_a = calc_stats(m_a, gt)
    stats_b = calc_stats(m_b, gt)
    stats_c = calc_stats(m_c, gt)
    stats_d = calc_stats(m_d, gt)

    avg_gru_latency = np.mean(gru_latencies_ms)
    gru_fps = 1000.0 / max(0.01, avg_gru_latency)

    print("\n" + "=" * 90)
    print(f"{'Method / Pipeline':<28}{'MAE (cm)':<12}{'RMSE (cm)':<12}{'MAPE (%)':<12}{'Std Dev (cm)':<14}{'Max Err (%)':<12}")
    print("-" * 90)
    print(f"{'A. D2 Only (Raw Math)':<28}{stats_a['mae_cm']:<12.2f}{stats_a['rmse_cm']:<12.2f}{stats_a['mape_pct']:<12.2f}{stats_a['std_cm']:<14.2f}{stats_a['max_err_pct']:<12.2f}")
    print(f"{'B. D2 + Kalman (Baseline)':<28}{stats_b['mae_cm']:<12.2f}{stats_b['rmse_cm']:<12.2f}{stats_b['mape_pct']:<12.2f}{stats_b['std_cm']:<14.2f}{stats_b['max_err_pct']:<12.2f}")
    print(f"{'C. D2 + GRU':<28}{stats_c['mae_cm']:<12.2f}{stats_c['rmse_cm']:<12.2f}{stats_c['mape_pct']:<12.2f}{stats_c['std_cm']:<14.2f}{stats_c['max_err_pct']:<12.2f}")
    print(f"{'D. D2 + GRU + Kalman (New)':<28}{stats_d['mae_cm']:<12.2f}{stats_d['rmse_cm']:<12.2f}{stats_d['mape_pct']:<12.2f}{stats_d['std_cm']:<14.2f}{stats_d['max_err_pct']:<12.2f}")
    print("=" * 90)
    print(f"[*] GRU Model Inference Latency: {avg_gru_latency:.3f} ms/frame ({gru_fps:.0f} FPS throughput)")
    print(f"[*] Baseline Error (D2 + Kalman):  {stats_b['mape_pct']:.2f}% ({stats_b['mae_cm']:.1f} cm)")
    print(f"[*] Proposed Error (D2+GRU+Kalman): {stats_d['mape_pct']:.2f}% ({stats_d['mae_cm']:.1f} cm)")
    
    error_reduction = stats_b['mape_pct'] - stats_d['mape_pct']
    if error_reduction > 0:
        rel_improvement = (error_reduction / stats_b['mape_pct']) * 100.0
        print(f"[+] Result: GRU Reduced Error by {error_reduction:.2f} percentage points ({rel_improvement:.1f}% relative reduction)")
    else:
        print(f"[-] Result: GRU did not reduce error (Error change: {error_reduction:+.2f} percentage points)")
    print("=" * 90)

    # Return summary dict for plotting & reporting
    return {
        "gt": gt,
        "d2_only": m_a,
        "d2_kalman": m_b,
        "d2_gru": m_c,
        "d2_gru_kalman": m_d,
        "stats": {
            "D2_only": stats_a,
            "D2_Kalman": stats_b,
            "D2_GRU": stats_c,
            "D2_GRU_Kalman": stats_d
        },
        "latency_ms": avg_gru_latency,
        "fps": gru_fps
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Distance Estimation Pipelines")
    parser.add_argument("--data", type=str, default="datasets/distance_sequences.json", help="Path to sequence dataset JSON")
    parser.add_argument("--model", type=str, default="models/distance_gru.pth", help="Path to GRU model")
    parser.add_argument("--scaler", type=str, default="models/distance_gru_scaler.json", help="Path to scaler JSON")
    parser.add_argument("--log-csv", type=str, default="flight_logs/gru_distance_evaluation.csv", help="Output CSV log")
    args = parser.parse_args()

    evaluate_pipelines(
        dataset_path=args.data,
        model_path=args.model,
        scaler_path=args.scaler,
        log_csv_path=args.log_csv
    )
