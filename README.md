# Never Let You Breathe

**Measuring the structural manipulation of attention in short-form video content.**

This project provides an open-source pipeline that analyzes YouTube Shorts (and potentially TikTok/Reels) for structural patterns associated with virality — not what content *says*, but how it's *constructed* to hold attention.

## Key Finding

Across a 95-video pilot study of 10 creators spanning left-wing, right-wing, and non-political content:

- Viral videos scored higher on **relentlessness** (sustained emotional activation with limited recovery) in **10 out of 10 creators**
- The pattern was **bipartisan** — structurally identical across political orientation
- Viral content spent **36% of runtime** in high-activation emotional states vs. **29%** for average-performing content from the same creators
- Viral content was often **slower**, not faster — relentlessness is pressure, not speed

The production study (500 videos, 30 channels) is in progress.

## What It Measures

| Metric | What It Captures |
|--------|-----------------|
| Emotional activation % | Time spent in fear, anger, sadness, disgust, surprise |
| Recovery period count | Drops to neutral/joy after sustained activation |
| Silence percentage | Actual silence in the audio track |
| Speaking pace (WPM) | Words per minute from Whisper transcription |
| Energy CV | Coefficient of variation in vocal energy (low = flatline high) |
| Pitch CV | Coefficient of variation in pitch contour |
| Cuts per minute | Visual scene changes per minute |
| **Relentlessness composite** | Weighted combination of activation, recovery suppression, silence absence, and energy stability |

## Pipeline Overview

```
YouTube Shorts
      │
      ▼
┌─────────────┐    ┌──────────────┐    ┌───────────────────┐
│ collect_     │───▶│ download_    │───▶│ analyze_videos.py │
│ videos.py   │    │ videos.sh    │    │                   │
│             │    │ (yt-dlp)     │    │ • Whisper STT     │
│ YouTube API │    │              │    │ • Emotion class.  │
│ metadata    │    │              │    │ • Prosody (F0/RMS)│
│             │    │              │    │ • Silence detect  │
│             │    │              │    │ • Cut detection   │
└─────────────┘    └──────────────┘    └───────┬───────────┘
                                               │
                                               ▼
                                    ┌──────────────────────┐
                                    │ Outputs:             │
                                    │ • full_analysis.json │
                                    │ • analysis_summary.csv│
                                    │ • paired_comparison  │
                                    │ • statistical tests  │
                                    └──────────────────────┘
```

## Quick Start

### Requirements

```bash
# System dependencies
sudo apt install ffmpeg

# Python packages
pip install openai-whisper librosa opencv-python-headless transformers torch scipy yt-dlp google-api-python-client
```

### 1. Collect Video Metadata

```bash
export YOUTUBE_API_KEY="your_key"
python3 collect_videos.py --output ./data/
```

### 2. Download Videos

```bash
bash data/download_videos.sh --cookies-from-browser firefox
```

### 3. Run Analysis

```bash
# Pilot (fast): uses whisper tiny model
python3 analyze_videos.py --input ./data/videos/ --metadata ./data/collected_videos.csv --output ./data/analysis/

# Production: edit CONFIG whisper_model to "medium" first
```

### 4. View Results

Results are in `data/analysis/`:
- `full_analysis.json` — Complete time-series data for every video
- `analysis_summary.csv` — One row per video, all metrics
- `paired_comparison.csv` — Within-creator viral vs. average comparisons

## Study Design

**Within-creator paired comparison**: For each creator, we compare their most viral Shorts to their average-performing Shorts. Same creator, same general style, same audience. The only systematic difference is platform performance.

**Mirrored political pairing**: Political channels are matched by format across lean:
- Solo commentators: right ↔ left
- Institutional news: right ↔ left  
- Instructional/educational: right ↔ left
- Populist/grassroots: right ↔ left

This design tests whether structural patterns are a property of virality itself rather than any particular ideology.

## Methodology Notes

- **Emotion classification**: Hartmann et al.'s `emotion-english-distilroberta-base` transformer model, applied to sliding 3-second windows over Whisper transcripts
- **Prosodic analysis**: F0 (pitch) and RMS (energy) contours via librosa, summarized as coefficient of variation
- **Cut detection**: Dual-method approach using ffprobe scene detection and OpenCV histogram correlation (handles AV1/VP9 codecs)
- **Relentlessness composite**: `0.3 × high_activation% + 0.25 × (100 - recovery%) + 0.25 × (100 - silence%) + 0.2 × (100 × (1 - energy_cv))`

The composite score is exploratory. All individual components are reported separately for transparency.

## Limitations

- Emotion classification is applied to *text*, not audio tone or visual content — it captures what is being said, not how it looks or sounds (prosody is measured separately)
- Whisper transcription quality varies; the production study uses the `medium` model
- View count as a proxy for virality conflates algorithmic promotion with organic sharing
- The pipeline does not measure visual manipulation (color grading, zoom dynamics, text overlays)
- Sample sizes in the pilot are small; the production study addresses this

## Project Status

| Phase | Status |
|-------|--------|
| Pilot study (95 videos, 10 channels) | ✅ Complete |
| Substack post #1 | ✅ Published |
| Open-source pipeline | ✅ Released |
| Production study (500 videos, 30 channels) | 🔄 In progress |
| Statistical significance testing | 🔄 In progress |
| Cuts per minute fix (AV1 codec) | ✅ Fixed |
| Breathe Score (public-facing 0-100) | 📋 Planned |
| Academic paper | 📋 Planned |
| Substack post #2 (production results) | 📋 Planned |

## Citation

If you use this pipeline in research, please cite:

```
@misc{neverletyoubreathe2026,
  title={Never Let You Breathe: Measuring Structural Manipulation in Short-Form Video},
  author={[Justin McGuigan]},
  year={2026},
  url={https://github.com/[jguigs]/never-let-you-breathe}
}
```

## License

MIT — use this however you want. The point is to make the mechanism visible.
