#!/usr/bin/env python3
"""
Generate Distance Estimation Evaluation Plots:
Generates high-resolution visualization plots comparing D2, D2+Kalman, D2+GRU, and D2+GRU+Kalman:
1. Actual distance vs D2 distance
2. Actual distance vs GRU-corrected distance
3. Error vs distance
4. Error distribution
5. D2 + Kalman vs GRU + Kalman

Saves all figures into flight_logs/
"""

import os
import sys
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt

from evaluate_gru_correction import evaluate_pipelines


def generate_plots(eval_results, out_dir="flight_logs"):
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    gt = eval_results["gt"]
    d2 = eval_results["d2_only"]
    d2_kalman = eval_results["d2_kalman"]
    d2_gru = eval_results["d2_gru"]
    d2_gru_kalman = eval_results["d2_gru_kalman"]
    stats = eval_results["stats"]

    # Style configuration
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    plt.rcParams.update({
        "font.sans-serif": "DejaVu Sans",
        "font.family": "sans-serif",
        "figure.titlesize": 14,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.dpi": 200
    })

    # 1. Actual distance vs D2 distance
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    ax1.scatter(gt, d2, alpha=0.6, color="#1f77b4", s=25, label="D2 Raw Estimation", edgecolors="none")
    min_d, max_d = min(np.min(gt), np.min(d2)) * 0.9, max(np.max(gt), np.max(d2)) * 1.05
    ax1.plot([min_d, max_d], [min_d, max_d], "r--", linewidth=1.8, label="Ideal 1:1 Ground Truth")
    ax1.set_xlabel("Actual Ground-Truth Distance (m)")
    ax1.set_ylabel("D2 Mathematical Distance (m)")
    ax1.set_title(f"Actual Distance vs D2 Distance (MAPE: {stats['D2_only']['mape_pct']:.2f}%)")
    ax1.legend(loc="upper left")
    ax1.set_xlim(min_d, max_d)
    ax1.set_ylim(min_d, max_d)
    plt.tight_layout()
    p1 = out_path / "actual_vs_d2_distance.png"
    fig1.savefig(p1)
    plt.close(fig1)

    # 2. Actual distance vs GRU-corrected distance
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    ax2.scatter(gt, d2_gru, alpha=0.6, color="#2ca02c", s=25, label="D2 + GRU Corrected", edgecolors="none")
    min_d, max_d = min(np.min(gt), np.min(d2_gru)) * 0.9, max(np.max(gt), np.max(d2_gru)) * 1.05
    ax2.plot([min_d, max_d], [min_d, max_d], "r--", linewidth=1.8, label="Ideal 1:1 Ground Truth")
    ax2.set_xlabel("Actual Ground-Truth Distance (m)")
    ax2.set_ylabel("GRU-Corrected Distance (m)")
    ax2.set_title(f"Actual Distance vs GRU-Corrected Distance (MAPE: {stats['D2_GRU']['mape_pct']:.2f}%)")
    ax2.legend(loc="upper left")
    ax2.set_xlim(min_d, max_d)
    ax2.set_ylim(min_d, max_d)
    plt.tight_layout()
    p2 = out_path / "actual_vs_gru_corrected_distance.png"
    fig2.savefig(p2)
    plt.close(fig2)

    # 3. Error vs distance
    fig3, ax3 = plt.subplots(figsize=(9, 6))
    err_d2 = np.abs(d2 - gt) * 100.0
    err_d2_k = np.abs(d2_kalman - gt) * 100.0
    err_gru_k = np.abs(d2_gru_kalman - gt) * 100.0

    ax3.scatter(gt, err_d2, alpha=0.35, color="#1f77b4", s=18, label=f"D2 Only (MAE: {stats['D2_only']['mae_cm']:.1f}cm)")
    ax3.scatter(gt, err_d2_k, alpha=0.45, color="#ff7f0e", s=18, label=f"D2 + Kalman (MAE: {stats['D2_Kalman']['mae_cm']:.1f}cm)")
    ax3.scatter(gt, err_gru_k, alpha=0.65, color="#2ca02c", s=22, label=f"D2 + GRU + Kalman (MAE: {stats['D2_GRU_Kalman']['mae_cm']:.1f}cm)")
    ax3.set_xlabel("Actual Ground-Truth Distance (m)")
    ax3.set_ylabel("Absolute Error (cm)")
    ax3.set_title("Absolute Estimation Error vs Target Distance")
    ax3.legend(loc="upper left")
    plt.tight_layout()
    p3 = out_path / "error_vs_distance.png"
    fig3.savefig(p3)
    plt.close(fig3)

    # 4. Error distribution
    fig4, ax4 = plt.subplots(figsize=(8, 6))
    raw_err_d2_k = (d2_kalman - gt) * 100.0
    raw_err_gru_k = (d2_gru_kalman - gt) * 100.0

    bins = np.linspace(-40, 40, 45)
    ax4.hist(raw_err_d2_k, bins=bins, alpha=0.55, color="#ff7f0e", label=f"D2 + Kalman (Std: {stats['D2_Kalman']['std_cm']:.1f}cm, MAPE: {stats['D2_Kalman']['mape_pct']:.2f}%)", density=True)
    ax4.hist(raw_err_gru_k, bins=bins, alpha=0.65, color="#2ca02c", label=f"D2 + GRU + Kalman (Std: {stats['D2_GRU_Kalman']['std_cm']:.1f}cm, MAPE: {stats['D2_GRU_Kalman']['mape_pct']:.2f}%)", density=True)
    ax4.axvline(0, color="black", linestyle="--", linewidth=1.2, alpha=0.7)
    ax4.set_xlabel("Signed Estimation Error (cm)")
    ax4.set_ylabel("Probability Density")
    ax4.set_title("Estimation Error Distribution (Baseline vs GRU-Kalman)")
    ax4.legend(loc="upper right")
    plt.tight_layout()
    p4 = out_path / "error_distribution.png"
    fig4.savefig(p4)
    plt.close(fig4)

    # 5. D2 + Kalman vs GRU + Kalman
    fig5, ax5 = plt.subplots(figsize=(9, 6))
    time_indices = np.arange(min(150, len(gt)))
    ax5.plot(time_indices, gt[:len(time_indices)], "k--", linewidth=2.0, label="Ground Truth Distance")
    ax5.plot(time_indices, d2_kalman[:len(time_indices)], color="#ff7f0e", linewidth=1.6, label=f"D2 + Kalman (Baseline, MAPE: {stats['D2_Kalman']['mape_pct']:.2f}%)")
    ax5.plot(time_indices, d2_gru_kalman[:len(time_indices)], color="#2ca02c", linewidth=1.8, label=f"D2 + GRU + Kalman (Proposed, MAPE: {stats['D2_GRU_Kalman']['mape_pct']:.2f}%)")
    ax5.set_xlabel("Evaluation Frame Step")
    ax5.set_ylabel("Estimated Distance (m)")
    ax5.set_title("Trajectory Tracking: D2 + Kalman vs D2 + GRU + Kalman")
    ax5.legend(loc="upper left")
    plt.tight_layout()
    p5 = out_path / "d2_kalman_vs_gru_kalman.png"
    fig5.savefig(p5)
    plt.close(fig5)

    print(f"[+] Successfully generated 5 benchmark plots in {out_dir}:")
    print(f"  1. {p1}")
    print(f"  2. {p2}")
    print(f"  3. {p3}")
    print(f"  4. {p4}")
    print(f"  5. {p5}")


def main():
    parser = argparse.ArgumentParser(description="Generate Distance Evaluation Plots")
    parser.add_argument("--data", type=str, default="datasets/distance_sequences.json", help="Path to sequence dataset JSON")
    parser.add_argument("--model", type=str, default="models/distance_gru.pth", help="Path to GRU model")
    parser.add_argument("--scaler", type=str, default="models/distance_gru_scaler.json", help="Path to scaler JSON")
    parser.add_argument("--out-dir", type=str, default="flight_logs", help="Directory to save plots")
    args = parser.parse_args()

    results = evaluate_pipelines(
        dataset_path=args.data,
        model_path=args.model,
        scaler_path=args.scaler
    )
    if results:
        generate_plots(results, out_dir=args.out_dir)


if __name__ == "__main__":
    main()
