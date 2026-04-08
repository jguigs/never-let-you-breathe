"""
Never Let You Breathe — Production Run Colab Notebook
=====================================================
Copy each section into a separate Colab cell.
Runtime: GPU → T4 (Runtime → Change runtime type → T4 GPU)

Estimated time: ~2.5 hours for 300 videos on T4 GPU.
"""

# ══════════════════════════════════════════════════════════════
# CELL 1: Install dependencies
# ══════════════════════════════════════════════════════════════
"""
!pip install -q openai-whisper librosa opencv-python-headless transformers torch scipy
!apt install -qq ffmpeg -y
"""

# ══════════════════════════════════════════════════════════════
# CELL 2: Mount Google Drive + verify GPU
# ══════════════════════════════════════════════════════════════
"""
from google.colab import drive
drive.mount('/content/drive')

import torch
assert torch.cuda.is_available(), "No GPU detected! Go to Runtime → Change runtime type → T4 GPU"
print(f"✓ GPU: {torch.cuda.get_device_name(0)}")

# Create working directories
!mkdir -p /content/drive/MyDrive/nlyb_production/analysis
!ls /content/drive/MyDrive/nlyb_production/videos/ | head -5
!echo "---"
!ls /content/drive/MyDrive/nlyb_production/videos/ | wc -l
!echo "video files found"
"""

# ══════════════════════════════════════════════════════════════
# CELL 3: Copy videos to local disk (faster I/O than Drive)
# ══════════════════════════════════════════════════════════════
"""
!mkdir -p /content/videos
!cp /content/drive/MyDrive/nlyb_production/videos/* /content/videos/
!echo "Copied $(ls /content/videos/ | wc -l) videos to local disk"
"""

# ══════════════════════════════════════════════════════════════
# CELL 4: Upload patched scripts + metadata
# ══════════════════════════════════════════════════════════════
"""
from google.colab import files
print("Upload: analyze_videos.py, collected_videos.csv")
uploaded = files.upload()
"""

# ══════════════════════════════════════════════════════════════
# CELL 5: Run analysis (output → Drive for persistence)
# ══════════════════════════════════════════════════════════════
"""
!python3 analyze_videos.py \
    --input /content/videos/ \
    --metadata collected_videos.csv \
    --output /content/drive/MyDrive/nlyb_production/analysis/
"""

# ══════════════════════════════════════════════════════════════
# CELL 6: Run statistical tests
# ══════════════════════════════════════════════════════════════
"""
# Upload statistical_tests.py first, or run inline:
!python3 statistical_tests.py \
    /content/drive/MyDrive/nlyb_production/analysis/analysis_summary.csv \
    /content/drive/MyDrive/nlyb_production/analysis/statistical_tests.json
"""

# ══════════════════════════════════════════════════════════════
# CELL 7: Quick results preview
# ══════════════════════════════════════════════════════════════
"""
import pandas as pd
import json

# Load summary
df = pd.read_csv('/content/drive/MyDrive/nlyb_production/analysis/analysis_summary.csv')
print(f"Total videos analyzed: {len(df)}")
print(f"Viral: {len(df[df.viral_status=='viral'])} | Average: {len(df[df.viral_status=='average'])}")
print()

# Quick comparison
viral = df[df.viral_status == 'viral']
avg = df[df.viral_status == 'average']

key_metrics = ['relentlessness_composite', 'high_activation_pct', 'recovery_pct',
               'silence_pct', 'cuts_per_minute', 'wpm', 'energy_cv']

print(f"{'Metric':<28} {'Viral':>8} {'Average':>8} {'Diff':>8}")
print("-" * 56)
for m in key_metrics:
    v = viral[m].mean()
    a = avg[m].mean()
    print(f"{m:<28} {v:>8.2f} {a:>8.2f} {v-a:>+8.2f}")

# Load stat tests
with open('/content/drive/MyDrive/nlyb_production/analysis/statistical_tests.json') as f:
    stats = json.load(f)
print(f"\nRelentlessness composite:")
r = stats['tests']['relentlessness_composite']
print(f"  Mann-Whitney p = {r['mann_whitney_p']:.4f} (adjusted: {r['mann_whitney_p_adjusted']:.4f})")
if 'wilcoxon_p' in r:
    print(f"  Wilcoxon p = {r['wilcoxon_p']:.4f} (adjusted: {r['wilcoxon_p_adjusted']:.4f})")
if 'cohens_d' in r:
    print(f"  Cohen's d = {r['cohens_d']:.3f} ({r['effect_size']})")
"""

# ══════════════════════════════════════════════════════════════
# CELL 8: Check cut detection fix (validation)
# ══════════════════════════════════════════════════════════════
"""
zero_cuts = df[df.cuts_per_minute == 0]
nonzero = df[df.cuts_per_minute > 0]
print(f"Cut detection results:")
print(f"  Zero cuts: {len(zero_cuts)} ({len(zero_cuts)/len(df)*100:.0f}%)")
print(f"  Nonzero:   {len(nonzero)} ({len(nonzero)/len(df)*100:.0f}%)")
print(f"  Mean cuts/min (nonzero): {nonzero.cuts_per_minute.mean():.1f}")
if len(zero_cuts)/len(df) > 0.5:
    print("  ⚠️  WARNING: >50% zero cuts — scene detection may still be failing on webm/AV1")
"""
