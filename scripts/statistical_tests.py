"""
Google Colab Production Run
============================
Copy this into a Colab notebook cell-by-cell.
Uses GPU runtime for Whisper medium model.

Runtime: ~2-3 hours for 300 videos on Colab T4 GPU.
"""

# ── CELL 1: Setup ──────────────────────────────────────────
# !pip install openai-whisper librosa opencv-python-headless transformers torch scipy yt-dlp
# !apt install ffmpeg -y

# ── CELL 2: Mount Drive (for persistent storage) ──────────
# from google.colab import drive
# drive.mount('/content/drive')
# !mkdir -p /content/drive/MyDrive/nlyb_production/videos
# !mkdir -p /content/drive/MyDrive/nlyb_production/analysis

# ── CELL 3: Upload your collected_videos.csv ──────────────
# from google.colab import files
# uploaded = files.upload()  # Upload collected_videos.csv

# ── CELL 4: Download videos (run in batches to avoid timeout) ──
# !bash download_videos.sh --cookies-from-browser firefox
# Note: On Colab, you may need to upload cookies.txt instead:
# !yt-dlp --cookies cookies.txt ...

# ── CELL 5: Run analysis with GPU ────────────────────────
# import torch
# print(f"GPU available: {torch.cuda.is_available()}")
# print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
#
# # In analyze_videos.py CONFIG, set:
# # "whisper_model": "medium"
# # And change device=-1 to device=0 in emotion classifier for GPU

# ── CELL 6: Statistical tests ────────────────────────────
"""
Run this after analysis completes to get significance tests.
"""

import csv
import numpy as np
from scipy import stats
from collections import defaultdict
import json


def run_statistical_tests(summary_csv_path, output_path=None):
    """
    Run comprehensive statistical tests on analysis results.
    
    Tests:
    1. Mann-Whitney U (aggregate viral vs average, unpaired)
    2. Wilcoxon signed-rank (within-creator paired)
    3. Sign test (direction consistency)
    4. Cohen's d (effect sizes)
    5. Bonferroni correction for multiple comparisons
    """
    
    # Load data
    rows = []
    with open(summary_csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    
    viral_rows = [r for r in rows if r['viral_status'] == 'viral']
    avg_rows = [r for r in rows if r['viral_status'] == 'average']
    
    metrics = [
        "cuts_per_minute", "silence_pct", "wpm",
        "high_activation_pct", "recovery_pct", "recovery_periods",
        "max_activation_streak", "energy_cv", "pitch_cv",
        "relentlessness_composite",
    ]
    
    results = {
        "n_viral": len(viral_rows),
        "n_average": len(avg_rows),
        "n_total": len(rows),
        "tests": {},
    }
    
    n_tests = len(metrics)  # For Bonferroni correction
    
    print(f"{'='*80}")
    print(f"STATISTICAL SIGNIFICANCE TESTS")
    print(f"Viral: {len(viral_rows)} videos | Average: {len(avg_rows)} videos")
    print(f"{'='*80}")
    
    # ── 1. Aggregate unpaired tests ──────────────────────────
    print(f"\n{'─'*80}")
    print(f"1. MANN-WHITNEY U (unpaired, aggregate)")
    print(f"{'─'*80}")
    print(f"{'Metric':<28} {'V Mean':>8} {'A Mean':>8} {'Diff':>8} {'U':>8} {'p':>8} {'p_adj':>8} {'Sig':>5}")
    
    for m in metrics:
        v_vals = [float(r[m]) for r in viral_rows]
        a_vals = [float(r[m]) for r in avg_rows]
        
        v_mean = np.mean(v_vals)
        a_mean = np.mean(a_vals)
        
        u_stat, p_val = stats.mannwhitneyu(v_vals, a_vals, alternative='two-sided')
        p_adj = min(p_val * n_tests, 1.0)  # Bonferroni
        
        sig = "***" if p_adj < 0.001 else "**" if p_adj < 0.01 else "*" if p_adj < 0.05 else "†" if p_adj < 0.1 else ""
        
        print(f"{m:<28} {v_mean:>8.2f} {a_mean:>8.2f} {v_mean-a_mean:>+8.2f} {u_stat:>8.0f} {p_val:>8.4f} {p_adj:>8.4f} {sig:>5}")
        
        results["tests"][m] = {
            "viral_mean": round(v_mean, 3),
            "average_mean": round(a_mean, 3),
            "diff": round(v_mean - a_mean, 3),
            "mann_whitney_U": float(u_stat),
            "mann_whitney_p": round(p_val, 6),
            "mann_whitney_p_adjusted": round(p_adj, 6),
        }
    
    # ── 2. Within-creator paired tests ───────────────────────
    print(f"\n{'─'*80}")
    print(f"2. WILCOXON SIGNED-RANK (within-creator paired)")
    print(f"{'─'*80}")
    
    # Build per-creator means
    by_creator = defaultdict(lambda: {"viral": [], "average": []})
    for row in rows:
        ch = row["channel"]
        status = row["viral_status"]
        if ch and status in ("viral", "average"):
            by_creator[ch][status].append(row)
    
    print(f"{'Metric':<28} {'Mean Δ':>8} {'W':>8} {'p':>8} {'p_adj':>8} {'Sig':>5} {'Dir':>14}")
    
    for m in metrics:
        diffs = []
        for ch, groups in by_creator.items():
            if groups["viral"] and groups["average"]:
                v_mean = np.mean([float(r[m]) for r in groups["viral"]])
                a_mean = np.mean([float(r[m]) for r in groups["average"]])
                diffs.append(v_mean - a_mean)
        
        mean_diff = np.mean(diffs)
        nonzero = [d for d in diffs if d != 0]
        
        if len(nonzero) >= 3:
            w_stat, p_val = stats.wilcoxon(nonzero, alternative='two-sided')
            p_adj = min(p_val * n_tests, 1.0)
            sig = "***" if p_adj < 0.001 else "**" if p_adj < 0.01 else "*" if p_adj < 0.05 else "†" if p_adj < 0.1 else ""
            direction = "viral ↑" if mean_diff > 0 else "avg ↑"
            print(f"{m:<28} {mean_diff:>+8.2f} {w_stat:>8.0f} {p_val:>8.4f} {p_adj:>8.4f} {sig:>5} {direction:>14}")
            
            results["tests"][m]["wilcoxon_W"] = float(w_stat)
            results["tests"][m]["wilcoxon_p"] = round(p_val, 6)
            results["tests"][m]["wilcoxon_p_adjusted"] = round(p_adj, 6)
            results["tests"][m]["n_creators_paired"] = len(diffs)
        else:
            print(f"{m:<28} {mean_diff:>+8.2f}    (insufficient non-zero pairs)")
    
    # ── 3. Effect sizes ──────────────────────────────────────
    print(f"\n{'─'*80}")
    print(f"3. EFFECT SIZES (Cohen's d)")
    print(f"{'─'*80}")
    
    for m in metrics:
        v_vals = [float(r[m]) for r in viral_rows]
        a_vals = [float(r[m]) for r in avg_rows]
        
        pooled_std = np.sqrt((np.std(v_vals, ddof=1)**2 + np.std(a_vals, ddof=1)**2) / 2)
        if pooled_std > 0:
            d = (np.mean(v_vals) - np.mean(a_vals)) / pooled_std
            size = "large" if abs(d) > 0.8 else "medium" if abs(d) > 0.5 else "small" if abs(d) > 0.2 else "negligible"
            print(f"  {m:<28} d = {d:>+.3f}  ({size})")
            results["tests"][m]["cohens_d"] = round(d, 4)
            results["tests"][m]["effect_size"] = size
    
    # ── 4. Sign test (consistency) ───────────────────────────
    print(f"\n{'─'*80}")
    print(f"4. SIGN TEST (direction consistency across creators)")
    print(f"{'─'*80}")
    
    for m in metrics:
        diffs = []
        for ch, groups in by_creator.items():
            if groups["viral"] and groups["average"]:
                v_mean = np.mean([float(r[m]) for r in groups["viral"]])
                a_mean = np.mean([float(r[m]) for r in groups["average"]])
                diffs.append(v_mean - a_mean)
        
        pos = sum(1 for d in diffs if d > 0)
        neg = sum(1 for d in diffs if d < 0)
        tied = sum(1 for d in diffs if d == 0)
        n = pos + neg
        
        if n > 0:
            p_sign = stats.binomtest(pos, n, 0.5).pvalue
            sig = "*" if p_sign < 0.05 else ""
            print(f"  {m:<28} {pos}/{len(diffs)} viral↑, {neg} avg↑, {tied} tied  (p={p_sign:.4f}) {sig}")
            results["tests"][m]["sign_test_viral_higher"] = pos
            results["tests"][m]["sign_test_n"] = len(diffs)
            results["tests"][m]["sign_test_p"] = round(p_sign, 6)
    
    # ── Save results ─────────────────────────────────────────
    if output_path:
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {output_path}")
    
    return results


if __name__ == "__main__":
    import sys
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "./data/analysis/analysis_summary.csv"
    output = sys.argv[2] if len(sys.argv) > 2 else "./data/analysis/statistical_tests.json"
    run_statistical_tests(csv_path, output)
