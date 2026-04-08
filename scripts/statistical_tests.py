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
        
        if not diffs:
            print(f"  {m:<28} — no valid paired creators, skipping")
            continue

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
            results["tests"][m]["sign_test_n"] = n
            results["tests"][m]["sign_test_p"] = round(p_sign, 6)
    
    # ── 5. Grouped analysis by political lean ─────────────────
    print(f"\n{'─'*80}")
    print(f"5. CREATOR-LEVEL DELTAS BY POLITICAL LEAN")
    print(f"{'─'*80}")

    # Build creator-level deltas with lean labels
    creator_deltas = {}  # {channel: {"lean": str, "deltas": {metric: float}}}
    for ch, groups in by_creator.items():
        if groups["viral"] and groups["average"]:
            # Get lean from the first viral row (consistent within creator)
            lean = groups["viral"][0].get("lean", "")
            deltas = {}
            for m in metrics:
                v_mean = np.mean([float(r[m]) for r in groups["viral"]])
                a_mean = np.mean([float(r[m]) for r in groups["average"]])
                deltas[m] = v_mean - a_mean
            creator_deltas[ch] = {"lean": lean, "deltas": deltas}

    # Group deltas by lean
    lean_groups = defaultdict(list)  # {lean: [creator_deltas_dict, ...]}
    for ch, info in creator_deltas.items():
        label = info["lean"] if info["lean"] else "non-political"
        lean_groups[label].append(info["deltas"])

    grouped_results = {}
    for label, delta_list in sorted(lean_groups.items()):
        grouped_results[label] = {"n_creators": len(delta_list), "metrics": {}}
        for m in metrics:
            vals = [d[m] for d in delta_list]
            pos = sum(1 for v in vals if v > 0)
            neg = sum(1 for v in vals if v < 0)
            zero = sum(1 for v in vals if v == 0)
            grouped_results[label]["metrics"][m] = {
                "n_creators": len(vals),
                "mean_delta": round(float(np.mean(vals)), 4),
                "median_delta": round(float(np.median(vals)), 4),
                "positive_count": pos,
                "negative_count": neg,
                "zero_count": zero,
            }

    # Print summary table for key metrics
    key_metrics = ["high_activation_pct", "recovery_pct", "relentlessness_composite", "cuts_per_minute"]
    group_labels = sorted(lean_groups.keys())

    print(f"\n  {'Metric':<28}", end="")
    for label in group_labels:
        n = len(lean_groups[label])
        print(f" {label+'('+str(n)+')':>18}", end="")
    print()

    for m in key_metrics:
        print(f"  {m:<28}", end="")
        for label in group_labels:
            info = grouped_results[label]["metrics"][m]
            print(f" {info['mean_delta']:>+8.2f} ({info['positive_count']}↑{info['negative_count']}↓)", end="")
        print()

    # Kruskal-Wallis across lean groups where sample sizes permit
    # Center (n=2) is too small for inferential testing — skip it
    kw_eligible_labels = [label for label in group_labels if len(lean_groups[label]) >= 3 and label != "center"]

    print(f"\n  Kruskal-Wallis across groups: {', '.join(kw_eligible_labels)}")
    if "center" in group_labels:
        print(f"  (center excluded: n={len(lean_groups['center'])}, too small for inferential test)")

    for m in metrics:
        kw_samples = [
            [d[m] for d in lean_groups[label]]
            for label in kw_eligible_labels
        ]
        # Need at least 2 groups with non-zero variance
        nonempty = [s for s in kw_samples if len(s) >= 3]

        if len(nonempty) >= 2:
            h_stat, p_val = stats.kruskal(*kw_samples)
            sig = "*" if p_val < 0.05 else ""
            print(f"  {m:<28} H={h_stat:>7.2f}  p={p_val:.4f} {sig}")
            for label in kw_eligible_labels:
                grouped_results[label]["metrics"][m]["kruskal_wallis_H"] = round(float(h_stat), 4)
                grouped_results[label]["metrics"][m]["kruskal_wallis_p"] = round(p_val, 6)
        else:
            print(f"  {m:<28} — skipped (fewer than 2 eligible groups with n≥3)")
            for label in kw_eligible_labels:
                grouped_results[label]["metrics"][m]["kruskal_wallis_skipped"] = "fewer than 2 eligible groups"

    results["grouped_by_lean"] = grouped_results

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
