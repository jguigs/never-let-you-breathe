"""
Structural Analysis Pipeline
==============================
Analyzes downloaded video files to extract:

STATIC METRICS (one number per video):
  - Cuts per minute (frame differencing)
  - Silence percentage (energy thresholding)
  - Speaking pace in WPM (Whisper transcription)
  - Music presence (binary) + tempo if present

TIME-SERIES (synchronized streams across video duration):
  - Emotional classification per transcript segment
  - Pitch contour (F0 frame-by-frame)
  - Energy contour (RMS frame-by-frame)

DERIVED MANIPULATION METRICS:
  - Emotional variance (coefficient of variation across segments)
  - Recovery period count (drops to neutral/low activation)
  - Sustained activation percentage (time above threshold)

Usage:
    python3 analyze_videos.py --input ./data/videos/ --metadata ./data/collected_videos.csv --output ./data/analysis/

Requires:
    pip install librosa openai-whisper opencv-python-headless transformers torch scipy
    ffmpeg must be installed (apt install ffmpeg)
"""

import os
import sys
import json
import csv
import argparse
import warnings
import numpy as np
from pathlib import Path
from datetime import datetime

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# GPU DETECTION + SINGLETON CLASSIFIER
# ─────────────────────────────────────────────────────────────

import torch as _torch
DEVICE = "cuda" if _torch.cuda.is_available() else "cpu"
DEVICE_IDX = 0 if DEVICE == "cuda" else -1
print(f"[INIT] Device: {DEVICE}" + (f" ({_torch.cuda.get_device_name(0)})" if DEVICE == "cuda" else ""))

_whisper_model = None
_emotion_classifier = None

def get_whisper_model():
    """Singleton: load Whisper once, reuse across all videos."""
    global _whisper_model
    if _whisper_model is None:
        import whisper
        print(f"    Loading Whisper '{CONFIG['whisper_model']}' on {DEVICE}...")
        _whisper_model = whisper.load_model(CONFIG["whisper_model"], device=DEVICE)
    return _whisper_model

# ─────────────────────────────────────────────────────────────
# SINGLETON CLASSIFIER (see get_emotion_classifier above)
# ─────────────────────────────────────────────────────────────

def get_emotion_classifier():
    """Singleton: load the emotion classifier once, reuse across all videos."""
    global _emotion_classifier
    if _emotion_classifier is None:
        from transformers import pipeline as _hf_pipeline
        print(f"    Loading emotion model on {DEVICE}...")
        _emotion_classifier = _hf_pipeline(
            "text-classification",
            model=CONFIG["emotion_model"],
            top_k=None,
            device=DEVICE_IDX,
        )
    return _emotion_classifier

# ─────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────

CONFIG = {
    # Cuts detection
    "cut_threshold": 30.0,          # Frame difference threshold (0-255 scale)
    "cut_min_interval_frames": 5,   # Minimum frames between cuts (avoids double-counting)

    # Silence detection
    "silence_threshold_db": -40,    # dB threshold below which audio is "silent"
    "silence_hop_length": 512,      # Audio analysis hop length

    # Whisper
    "whisper_model": "medium",       # PRODUCTION: "medium" for accuracy (was "tiny" in pilot)

    # Emotion classifier
    "emotion_model": "j-hartmann/emotion-english-distilroberta-base",
    # Outputs: anger, disgust, fear, joy, neutral, sadness, surprise

    # Emotion time-series
    "emotion_window_seconds": 3.0,  # Sliding window size for emotion classification
    "emotion_step_seconds": 1.0,    # Step size between windows

    # Prosodic analysis
    "prosody_hop_length": 512,      # Hop length for pitch/energy extraction
    "prosody_sr": 22050,            # Sample rate for audio analysis

    # Recovery period detection
    "recovery_emotions": ["neutral", "joy"],  # Emotions that count as "recovery"
    "high_activation_emotions": ["anger", "fear", "sadness", "disgust", "surprise"],

    # Sustained activation
    "energy_high_percentile": 70,   # Percentile above which energy is "high"
    "sustained_activation_min_seconds": 3.0,  # One full classification window; activation must persist this long to count as sustained
    "recovery_min_seconds": 2.0,  # Lower than activation threshold because recovery intervals in short-form video are naturally shorter, but filters single-window noise

    # Robustness: alternative window configs to bracket the primary (3.0, 1.0)
    "robustness_window_configs": [(2.0, 1.0), (4.0, 1.0)],
}


# ─────────────────────────────────────────────────────────────
# STATIC METRIC: CUTS PER MINUTE
# ─────────────────────────────────────────────────────────────

def analyze_cuts(video_path):
    """
    Detect scene cuts using multiple methods for robustness across codecs.

    The pilot study returned near-zero cuts because AV1/VP9 codecs in .webm
    files produce inter-frame compression artifacts that reduce mean pixel
    differences below the fixed threshold. This version uses:

    1. Histogram correlation (robust to compression artifacts)
    2. Adaptive thresholding (per-video, not a fixed number)
    3. Fallback to ffprobe scene detection if OpenCV reads few frames

    Returns: cuts_per_minute, total_cuts, duration_seconds, cut_timestamps, method
    """
    import cv2
    import subprocess

    # --- Method 1: ffmpeg scene detection (works on AV1/VP9 webm) ---
    try:
        import re as _re
        cmd = [
            "ffmpeg", "-i", str(video_path),
            "-vf", "select='gt(scene,0.3)',metadata=print:file=/dev/stdout",
            "-an", "-f", "null", "-",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        # pts_time appears on "frame:N  pts:...  pts_time:VALUE" lines in stdout
        timestamps = []
        for line in result.stdout.split("\n"):
            m = _re.search(r"pts_time:([\d.]+)", line)
            if m:
                timestamps.append(float(m.group(1)))

        if timestamps:
            # Get duration
            dur_cmd = [
                "ffprobe", "-v", "quiet",
                "-show_entries", "format=duration",
                "-of", "csv=p=0",
                str(video_path),
            ]
            dur_result = subprocess.run(dur_cmd, capture_output=True, text=True, timeout=10)
            duration = float(dur_result.stdout.strip()) if dur_result.stdout.strip() else 0

            if duration > 0:
                # Filter out cuts too close together
                min_interval = 0.15  # seconds
                filtered = [timestamps[0]]
                for t in timestamps[1:]:
                    if t - filtered[-1] >= min_interval:
                        filtered.append(t)

                cuts_per_minute = (len(filtered) / duration) * 60

                return {
                    "cuts_per_minute": round(cuts_per_minute, 2),
                    "total_cuts": len(filtered),
                    "duration_seconds": round(duration, 2),
                    "cut_timestamps": [round(t, 2) for t in filtered],
                    "detection_method": "ffmpeg_scene",
                }
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
        pass  # Fall through to OpenCV method

    # --- Method 2: OpenCV histogram correlation (handles AV1/VP9 better) ---
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0

    if duration == 0:
        cap.release()
        return None

    prev_hist = None
    correlations = []
    frame_idx = 0

    # Early bail-out: if the first frame can't be read (e.g. AV1 decode failure),
    # skip the entire OpenCV path rather than looping through an empty capture
    ret, first_frame = cap.read()
    if not ret:
        cap.release()
        return {
            "cuts_per_minute": 0,
            "total_cuts": 0,
            "duration_seconds": round(duration, 2),
            "cut_timestamps": [],
            "detection_method": "opencv_failed",
        }

    # Process first frame
    hsv = cv2.cvtColor(first_frame, cv2.COLOR_BGR2HSV)
    small = cv2.resize(hsv, (160, 90))
    prev_hist = cv2.calcHist([small], [0, 1], None, [50, 60], [0, 180, 0, 256])
    cv2.normalize(prev_hist, prev_hist)
    frame_idx = 1

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to HSV for better histogram comparison
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        small = cv2.resize(hsv, (160, 90))

        # Calculate histogram (Hue + Saturation channels)
        hist = cv2.calcHist([small], [0, 1], None, [50, 60], [0, 180, 0, 256])
        cv2.normalize(hist, hist)

        if prev_hist is not None:
            # Correlation: 1.0 = identical, 0.0 = completely different
            corr = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL)
            correlations.append((frame_idx, corr))

        prev_hist = hist.copy()
        frame_idx += 1

    cap.release()

    if not correlations:
        return {
            "cuts_per_minute": 0,
            "total_cuts": 0,
            "duration_seconds": round(duration, 2),
            "cut_timestamps": [],
            "detection_method": "histogram_adaptive",
        }

    # Adaptive threshold: cuts are frames where correlation drops significantly
    corr_values = [c[1] for c in correlations]
    median_corr = np.median(corr_values)
    std_corr = np.std(corr_values)

    # A cut is a frame where correlation drops below median - 2*std
    # but also below an absolute ceiling of 0.7 (to avoid false positives
    # on very static videos)
    threshold = min(median_corr - 2.0 * std_corr, 0.7)

    cuts = []
    min_interval_frames = max(5, int(fps * 0.15))  # At least 150ms between cuts
    last_cut_frame = -min_interval_frames

    for frame_idx, corr in correlations:
        if corr < threshold and (frame_idx - last_cut_frame) >= min_interval_frames:
            cuts.append(frame_idx / fps)
            last_cut_frame = frame_idx

    cuts_per_minute = (len(cuts) / duration) * 60 if duration > 0 else 0

    return {
        "cuts_per_minute": round(cuts_per_minute, 2),
        "total_cuts": len(cuts),
        "duration_seconds": round(duration, 2),
        "cut_timestamps": [round(t, 2) for t in cuts],
        "detection_method": "histogram_adaptive",
        "threshold_used": round(threshold, 4),
        "median_correlation": round(median_corr, 4),
    }


# ─────────────────────────────────────────────────────────────
# STATIC METRIC: SILENCE PERCENTAGE
# ─────────────────────────────────────────────────────────────

def analyze_silence(audio_path):
    """
    Calculate percentage of audio that is silence.
    Returns: silence_percentage, total_duration, silence_segments
    """
    import librosa

    y, sr = librosa.load(str(audio_path), sr=CONFIG["prosody_sr"])
    duration = len(y) / sr

    if duration == 0:
        return None

    # Calculate RMS energy in short frames
    hop = CONFIG["silence_hop_length"]
    rms = librosa.feature.rms(y=y, hop_length=hop)[0]
    rms_db = librosa.amplitude_to_db(rms, ref=np.max)

    # Count frames below silence threshold
    silent_frames = np.sum(rms_db < CONFIG["silence_threshold_db"])
    total_frames = len(rms_db)
    silence_pct = (silent_frames / total_frames) * 100 if total_frames > 0 else 0

    # Find silence segments (for time-series)
    frame_duration = hop / sr
    silence_mask = rms_db < CONFIG["silence_threshold_db"]
    segments = []
    in_silence = False
    start = 0

    for i, is_silent in enumerate(silence_mask):
        if is_silent and not in_silence:
            start = i * frame_duration
            in_silence = True
        elif not is_silent and in_silence:
            end = i * frame_duration
            if (end - start) > 0.1:  # Minimum 100ms to count
                segments.append({"start": round(start, 2), "end": round(end, 2)})
            in_silence = False

    if in_silence:
        end = len(silence_mask) * frame_duration
        if (end - start) > 0.1:
            segments.append({"start": round(start, 2), "end": round(end, 2)})

    return {
        "silence_percentage": round(silence_pct, 2),
        "silence_segments": segments,
        "total_duration": round(duration, 2),
    }


# ─────────────────────────────────────────────────────────────
# STATIC METRIC: SPEAKING PACE (WPM)
# ─────────────────────────────────────────────────────────────

def analyze_speech(audio_path):
    """
    Transcribe audio with Whisper and calculate speaking pace.
    Returns: wpm, word_count, transcript, word_timestamps
    """
    model = get_whisper_model()
    result = model.transcribe(
        str(audio_path),
        word_timestamps=True,
        language="en",
    )

    # Extract words with timestamps
    words = []
    for segment in result.get("segments", []):
        for word_info in segment.get("words", []):
            words.append({
                "word": word_info["word"].strip(),
                "start": round(word_info["start"], 2),
                "end": round(word_info["end"], 2),
            })

    # Calculate speaking pace
    if not words:
        return {
            "wpm": 0,
            "word_count": 0,
            "transcript": "",
            "word_timestamps": [],
        }

    total_words = len(words)
    # Speaking duration = time from first word to last word
    speaking_duration = words[-1]["end"] - words[0]["start"]
    wpm = (total_words / speaking_duration) * 60 if speaking_duration > 0 else 0

    full_transcript = result.get("text", "").strip()

    return {
        "wpm": round(wpm, 1),
        "word_count": total_words,
        "transcript": full_transcript,
        "word_timestamps": words,
    }


# ─────────────────────────────────────────────────────────────
# STATIC METRIC: MUSIC PRESENCE + TEMPO
# ─────────────────────────────────────────────────────────────

def analyze_music(audio_path):
    """
    Detect music presence and estimate tempo.
    Uses onset strength and spectral features to estimate
    whether background music is present.
    """
    import librosa

    y, sr = librosa.load(str(audio_path), sr=CONFIG["prosody_sr"])

    # Estimate tempo
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    # Handle both old and new librosa return types
    if isinstance(tempo, np.ndarray):
        tempo = float(tempo[0])
    else:
        tempo = float(tempo)

    # Onset strength as proxy for rhythmic content
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    onset_std = float(np.std(onset_env))

    # Spectral flatness — lower means more tonal (music-like)
    flatness = librosa.feature.spectral_flatness(y=y)
    mean_flatness = float(np.mean(flatness))

    # Heuristic: music is likely if there's rhythmic regularity
    # and spectral tonality
    has_music = len(beat_frames) > 4 and mean_flatness < 0.3

    return {
        "music_detected": has_music,
        "estimated_tempo_bpm": round(tempo, 1),
        "beat_count": len(beat_frames),
        "spectral_flatness": round(mean_flatness, 4),
        "onset_strength_std": round(onset_std, 4),
    }


# ─────────────────────────────────────────────────────────────
# TIME-SERIES: EMOTIONAL SEQUENCING
# ─────────────────────────────────────────────────────────────

def analyze_emotions(transcript_data, window_seconds=None, step_seconds=None):
    """
    Run emotion classification on sliding windows of the transcript.
    Returns time-series of emotion labels and scores.

    Optional window_seconds/step_seconds override CONFIG defaults (used for
    robustness checks across alternative window sizes).
    """
    if not transcript_data["word_timestamps"]:
        return {"emotion_timeseries": [], "dominant_emotions": {}}

    classifier = get_emotion_classifier()

    words = transcript_data["word_timestamps"]
    window_size = window_seconds if window_seconds is not None else CONFIG["emotion_window_seconds"]
    step_size = step_seconds if step_seconds is not None else CONFIG["emotion_step_seconds"]

    # Determine time range
    start_time = words[0]["start"]
    end_time = words[-1]["end"]

    timeseries = []
    current = start_time

    while current < end_time:
        window_end = current + window_size

        # Collect words in this window
        window_words = [
            w["word"] for w in words
            if w["start"] >= current and w["start"] < window_end
        ]

        if window_words:
            text = " ".join(window_words)
            try:
                result = classifier(text[:512])[0]  # Truncate to model max
                # Sort by score
                result.sort(key=lambda x: x["score"], reverse=True)
                top_emotion = result[0]["label"]
                scores = {r["label"]: round(r["score"], 4) for r in result}
            except Exception:
                top_emotion = "unknown"
                scores = {}

            timeseries.append({
                "time_start": round(current, 2),
                "time_end": round(window_end, 2),
                "text": text,
                "dominant_emotion": top_emotion,
                "scores": scores,
            })

        current += step_size

    # Aggregate: count dominant emotions
    emotion_counts = {}
    for entry in timeseries:
        e = entry["dominant_emotion"]
        emotion_counts[e] = emotion_counts.get(e, 0) + 1

    total = len(timeseries)
    emotion_distribution = {
        k: round(v / total * 100, 1)
        for k, v in emotion_counts.items()
    } if total > 0 else {}

    # Build contiguous emotion spans by merging adjacent/overlapping windows
    # with the same dominant emotion. When emotions differ between overlapping
    # windows, clip the new span's start to the previous span's end so that
    # spans tile the timeline without overlaps.
    spans = []
    for entry in timeseries:
        emotion = entry["dominant_emotion"]
        t_start = entry["time_start"]
        t_end = entry["time_end"]
        if spans and spans[-1]["emotion"] == emotion:
            # Extend the current span to the end of this window
            spans[-1]["end"] = t_end
            spans[-1]["duration_seconds"] = round(spans[-1]["end"] - spans[-1]["start"], 2)
        else:
            # Clip start to previous span's end to avoid overlap
            if spans and t_start < spans[-1]["end"]:
                t_start = spans[-1]["end"]
            spans.append({
                "emotion": emotion,
                "start": t_start,
                "end": t_end,
                "duration_seconds": round(t_end - t_start, 2),
            })

    # speech_covered_seconds = distance from first window start to last window end
    if timeseries:
        speech_covered_seconds = round(timeseries[-1]["time_end"] - timeseries[0]["time_start"], 2)
    else:
        speech_covered_seconds = 0

    return {
        "emotion_timeseries": timeseries,
        "emotion_spans": spans,
        "speech_covered_seconds": speech_covered_seconds,
        "emotion_distribution": emotion_distribution,
        "total_windows": total,
    }


# ─────────────────────────────────────────────────────────────
# TIME-SERIES: PROSODIC CONTOUR (PITCH + ENERGY)
# ─────────────────────────────────────────────────────────────

def analyze_prosody(audio_path):
    """
    Extract pitch (F0) and energy (RMS) contours over time.
    Returns frame-by-frame time-series and summary statistics.
    """
    import librosa

    y, sr = librosa.load(str(audio_path), sr=CONFIG["prosody_sr"])
    hop = CONFIG["prosody_hop_length"]

    # RMS energy
    rms = librosa.feature.rms(y=y, hop_length=hop)[0]
    rms_db = librosa.amplitude_to_db(rms, ref=np.max)

    # Pitch (F0) using pyin
    f0, voiced_flag, voiced_prob = librosa.pyin(
        y, fmin=60, fmax=500,
        sr=sr, hop_length=hop,
    )

    # Time axis
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop)

    # Clean F0 (replace NaN with 0 for unvoiced frames)
    f0_clean = np.nan_to_num(f0[:len(times)], nan=0.0)

    # Summary statistics
    voiced_f0 = f0_clean[f0_clean > 0]

    energy_stats = {
        "mean_db": round(float(np.mean(rms_db)), 2),
        "std_db": round(float(np.std(rms_db)), 2),
        "cv_energy": round(float(np.std(rms_db) / abs(np.mean(rms_db))) if np.mean(rms_db) != 0 else 0, 4),
        "pct_above_threshold": round(
            float(np.sum(rms_db > np.percentile(rms_db, CONFIG["energy_high_percentile"])) / len(rms_db) * 100), 1
        ),
    }

    pitch_stats = {
        "mean_f0": round(float(np.mean(voiced_f0)), 2) if len(voiced_f0) > 0 else 0,
        "std_f0": round(float(np.std(voiced_f0)), 2) if len(voiced_f0) > 0 else 0,
        "cv_pitch": round(
            float(np.std(voiced_f0) / np.mean(voiced_f0)) if len(voiced_f0) > 0 and np.mean(voiced_f0) > 0 else 0, 4
        ),
        "voiced_percentage": round(float(np.sum(f0_clean > 0) / len(f0_clean) * 100), 1) if len(f0_clean) > 0 else 0,
    }

    # Downsample time-series for storage (one point per 0.5s)
    target_interval = 0.5
    frame_interval = hop / sr
    downsample_factor = max(1, int(target_interval / frame_interval))

    timeseries = []
    for i in range(0, min(len(times), len(rms_db), len(f0_clean)), downsample_factor):
        timeseries.append({
            "time": round(float(times[i]), 2),
            "energy_db": round(float(rms_db[i]), 2),
            "f0_hz": round(float(f0_clean[i]), 1),
        })

    return {
        "energy_stats": energy_stats,
        "pitch_stats": pitch_stats,
        "prosody_timeseries": timeseries,
    }


# ─────────────────────────────────────────────────────────────
# DERIVED: MANIPULATION SIGNATURE METRICS
# ─────────────────────────────────────────────────────────────

def compute_manipulation_metrics(emotion_data, prosody_data, silence_data):
    """
    Compute the key manipulation signature metrics:
    - Emotional variance (do emotions change or flatline?)
    - Recovery periods (drops to neutral/positive)
    - Sustained activation percentage
    """
    metrics = {}

    # --- Emotional variance ---
    if emotion_data["emotion_timeseries"]:
        emotions = [e["dominant_emotion"] for e in emotion_data["emotion_timeseries"]]
        total = len(emotions)

        # Count high-activation vs recovery emotions
        high_activation_count = sum(
            1 for e in emotions if e in CONFIG["high_activation_emotions"]
        )
        recovery_count = sum(
            1 for e in emotions if e in CONFIG["recovery_emotions"]
        )

        metrics["high_activation_pct"] = round(high_activation_count / total * 100, 1) if total > 0 else 0
        metrics["recovery_pct"] = round(recovery_count / total * 100, 1) if total > 0 else 0

        # Count transitions to recovery (actual recovery periods)
        recovery_periods = 0
        in_activation = False
        for e in emotions:
            if e in CONFIG["high_activation_emotions"]:
                in_activation = True
            elif e in CONFIG["recovery_emotions"] and in_activation:
                recovery_periods += 1
                in_activation = False

        metrics["recovery_period_count"] = recovery_periods

        # Emotional diversity (how many distinct emotions appear)
        unique_emotions = len(set(emotions))
        metrics["emotional_diversity"] = unique_emotions

        # Longest streak of high-activation emotions
        max_streak = 0
        current_streak = 0
        for e in emotions:
            if e in CONFIG["high_activation_emotions"]:
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 0
        metrics["max_activation_streak"] = max_streak

        # Duration-based metrics from emotion spans
        spans = emotion_data.get("emotion_spans", [])
        speech_covered = emotion_data.get("speech_covered_seconds", 0)

        high_activation_seconds = sum(
            s["duration_seconds"] for s in spans
            if s["emotion"] in CONFIG["high_activation_emotions"]
        )
        recovery_seconds = sum(
            s["duration_seconds"] for s in spans
            if s["emotion"] in CONFIG["recovery_emotions"]
        )

        metrics["speech_covered_seconds"] = speech_covered
        metrics["high_activation_seconds"] = round(high_activation_seconds, 2)
        metrics["recovery_seconds"] = round(recovery_seconds, 2)
        metrics["high_activation_time_pct"] = round(
            high_activation_seconds / speech_covered * 100, 1
        ) if speech_covered > 0 else 0
        metrics["recovery_time_pct"] = round(
            recovery_seconds / speech_covered * 100, 1
        ) if speech_covered > 0 else 0

        # Sustained activation bouts: high-activation spans >= minimum duration
        min_sustained = CONFIG["sustained_activation_min_seconds"]
        activation_spans = [
            s for s in spans
            if s["emotion"] in CONFIG["high_activation_emotions"]
        ]
        sustained_bouts = [s for s in activation_spans if s["duration_seconds"] >= min_sustained]
        sustained_seconds = sum(s["duration_seconds"] for s in sustained_bouts)
        longest_bout = max((s["duration_seconds"] for s in activation_spans), default=0)

        metrics["sustained_activation_bout_count"] = len(sustained_bouts)
        metrics["sustained_activation_seconds"] = round(sustained_seconds, 2)
        metrics["sustained_activation_time_pct"] = round(
            sustained_seconds / speech_covered * 100, 1
        ) if speech_covered > 0 else 0
        metrics["longest_activation_bout_seconds"] = round(longest_bout, 2)

        # Duration-qualified recovery bouts: recovery spans >= minimum duration
        min_recovery = CONFIG["recovery_min_seconds"]
        recovery_spans = [
            s for s in spans
            if s["emotion"] in CONFIG["recovery_emotions"]
        ]
        true_recovery_bouts = [s for s in recovery_spans if s["duration_seconds"] >= min_recovery]
        true_recovery_secs = sum(s["duration_seconds"] for s in true_recovery_bouts)
        longest_recovery = max((s["duration_seconds"] for s in recovery_spans), default=0)

        metrics["true_recovery_bout_count"] = len(true_recovery_bouts)
        metrics["true_recovery_seconds"] = round(true_recovery_secs, 2)
        metrics["true_recovery_time_pct"] = round(
            true_recovery_secs / speech_covered * 100, 1
        ) if speech_covered > 0 else 0
        metrics["mean_recovery_bout_seconds"] = round(
            true_recovery_secs / len(true_recovery_bouts), 2
        ) if true_recovery_bouts else 0
        metrics["longest_recovery_bout_seconds"] = round(longest_recovery, 2)

    else:
        metrics["high_activation_pct"] = 0
        metrics["recovery_pct"] = 0
        metrics["recovery_period_count"] = 0
        metrics["emotional_diversity"] = 0
        metrics["max_activation_streak"] = 0
        metrics["speech_covered_seconds"] = 0
        metrics["high_activation_seconds"] = 0
        metrics["recovery_seconds"] = 0
        metrics["high_activation_time_pct"] = 0
        metrics["recovery_time_pct"] = 0
        metrics["sustained_activation_bout_count"] = 0
        metrics["sustained_activation_seconds"] = 0
        metrics["sustained_activation_time_pct"] = 0
        metrics["longest_activation_bout_seconds"] = 0
        metrics["true_recovery_bout_count"] = 0
        metrics["true_recovery_seconds"] = 0
        metrics["true_recovery_time_pct"] = 0
        metrics["mean_recovery_bout_seconds"] = 0
        metrics["longest_recovery_bout_seconds"] = 0

    # --- Prosodic sustained activation ---
    if prosody_data.get("energy_stats"):
        metrics["energy_cv"] = prosody_data["energy_stats"]["cv_energy"]
        metrics["pitch_cv"] = prosody_data["pitch_stats"]["cv_pitch"]
        # Low CV = flatline = sustained = more relentless
        # High CV = variable = peaks and valleys = more breathing room
    else:
        metrics["energy_cv"] = 0
        metrics["pitch_cv"] = 0

    # --- Silence as recovery ---
    if silence_data:
        metrics["silence_pct"] = silence_data["silence_percentage"]
    else:
        metrics["silence_pct"] = 0

    # --- Composite relentlessness score (experimental) ---
    # Higher = more relentless. Normalized 0-100.
    # Components: high activation %, inverse of recovery %, inverse of silence %,
    # inverse of energy CV (low variance = flatline high)
    # This is exploratory — report components individually in the paper,
    # use composite only for the public scorecard.
    ha = metrics.get("high_activation_time_pct", metrics["high_activation_pct"])
    rec_inv = 100 - metrics.get("true_recovery_time_pct", metrics["recovery_pct"])
    sil_inv = 100 - metrics["silence_pct"]
    # Normalize energy CV (typical range 0-1, invert so low CV = high score)
    ecv_score = max(0, min(100, (1 - metrics["energy_cv"]) * 100))

    metrics["relentlessness_composite"] = round(
        (ha * 0.3 + rec_inv * 0.25 + sil_inv * 0.25 + ecv_score * 0.2), 1
    )

    # --- Named subconstruct components (interpretive groupings of existing metrics) ---

    # Activation pressure component: how much speech time is high-activation,
    # and how much of that is sustained (not brief spikes)
    ha_time = metrics.get("high_activation_time_pct", 0)
    sa_time = metrics.get("sustained_activation_time_pct", 0)
    metrics["activation_pressure"] = round((ha_time + sa_time) / 2, 1)

    # Recovery scarcity component: how little recovery time the viewer gets.
    # Higher = less recovery. Bout count reported separately (not normalized).
    tr_time = metrics.get("true_recovery_time_pct", 0)
    metrics["recovery_scarcity"] = round(100 - tr_time, 1)

    # Space compression component: how little silence and how little vocal
    # variation exist — captures the absence of breathing room
    metrics["space_compression"] = round((sil_inv + ecv_score) / 2, 1)

    return metrics


# ─────────────────────────────────────────────────────────────
# AUDIO EXTRACTION
# ─────────────────────────────────────────────────────────────

def extract_audio(video_path, output_path):
    """Extract audio from video using ffmpeg."""
    import subprocess
    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-vn", "-acodec", "pcm_s16le", "-ar", "22050", "-ac", "1",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0


# ─────────────────────────────────────────────────────────────
# MAIN ANALYSIS PIPELINE
# ─────────────────────────────────────────────────────────────

def analyze_single_video(video_path, temp_dir):
    """Run full analysis pipeline on a single video file."""
    video_path = Path(video_path)
    audio_path = Path(temp_dir) / f"{video_path.stem}.wav"

    results = {
        "filename": video_path.name,
        "analysis_timestamp": datetime.now().isoformat(),
    }

    # Step 1: Extract audio
    print(f"    Extracting audio...")
    if not extract_audio(video_path, audio_path):
        print(f"    ERROR: Could not extract audio from {video_path}")
        return results

    # Step 2: Cuts per minute
    print(f"    Analyzing cuts...")
    cuts_data = analyze_cuts(video_path)
    if cuts_data:
        results["cuts"] = cuts_data

    # Step 3: Silence
    print(f"    Analyzing silence...")
    silence_data = analyze_silence(audio_path)
    if silence_data:
        results["silence"] = silence_data

    # Step 4: Speech / transcription
    print(f"    Transcribing speech...")
    speech_data = analyze_speech(audio_path)
    results["speech"] = speech_data

    # Step 5: Music detection
    print(f"    Analyzing music...")
    music_data = analyze_music(audio_path)
    results["music"] = music_data

    # Step 6: Emotional sequencing
    print(f"    Analyzing emotional sequences...")
    emotion_data = analyze_emotions(speech_data)
    results["emotions"] = emotion_data

    # Step 7: Prosodic contour
    print(f"    Analyzing prosody...")
    prosody_data = analyze_prosody(audio_path)
    results["prosody"] = prosody_data

    # Step 8: Manipulation signature metrics
    print(f"    Computing manipulation metrics...")
    manipulation_data = compute_manipulation_metrics(
        emotion_data, prosody_data, silence_data
    )
    results["manipulation_metrics"] = manipulation_data

    # Denominator fields: separate speech-covered time from full-audio time
    full_audio = silence_data["total_duration"] if silence_data else 0
    speech_covered = emotion_data.get("speech_covered_seconds", 0)
    results["denominators"] = {
        "full_audio_seconds": full_audio,
        "speech_covered_seconds": speech_covered,
        "non_speech_seconds": round(full_audio - speech_covered, 2),
        "speech_density_pct": round(
            speech_covered / full_audio * 100, 1
        ) if full_audio > 0 else 0,
    }

    # Step 9: Robustness checks — re-run emotion windowing with alternative configs
    # Reuses transcript (speech_data), prosody, and silence from the primary run
    robustness_configs = CONFIG.get("robustness_window_configs", [])
    if robustness_configs and speech_data.get("word_timestamps"):
        robustness = {}
        for win_sec, step_sec in robustness_configs:
            config_key = f"{win_sec}s_{step_sec}s"
            print(f"    Robustness check: {config_key}...")
            rob_emotion = analyze_emotions(speech_data, window_seconds=win_sec, step_seconds=step_sec)
            rob_metrics = compute_manipulation_metrics(rob_emotion, prosody_data, silence_data)
            robustness[config_key] = {
                "high_activation_time_pct": rob_metrics.get("high_activation_time_pct", 0),
                "recovery_time_pct": rob_metrics.get("recovery_time_pct", 0),
                "sustained_activation_time_pct": rob_metrics.get("sustained_activation_time_pct", 0),
                "true_recovery_time_pct": rob_metrics.get("true_recovery_time_pct", 0),
                "relentlessness_composite": rob_metrics.get("relentlessness_composite", 0),
            }
        results["robustness"] = robustness

    # Cleanup temp audio
    if audio_path.exists():
        audio_path.unlink()

    return results


def run_pipeline(input_dir, metadata_csv, output_dir):
    """Run the full pipeline across all videos."""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    temp_dir = output_dir / "temp"
    temp_dir.mkdir(exist_ok=True)

    # Load metadata to match filenames to video info
    metadata = {}
    if metadata_csv and Path(metadata_csv).exists():
        with open(metadata_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Match on video_id which appears in the filename
                metadata[row["video_id"]] = row

    # Find video files
    video_extensions = {".mp4", ".webm", ".mkv", ".avi", ".mov"}
    video_files = sorted([
        f for f in input_dir.iterdir()
        if f.suffix.lower() in video_extensions
    ])

    if not video_files:
        print(f"No video files found in {input_dir}")
        print(f"Run the download script first: bash data/download_videos.sh")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f" STRUCTURAL ANALYSIS PIPELINE")
    print(f" Videos: {len(video_files)}")
    print(f" Output: {output_dir}")
    print(f"{'='*60}\n")

    all_results = []
    summary_rows = []

    # ── CHECKPOINT SUPPORT ────────────────────────────────────
    checkpoint_path = output_dir / "checkpoint.json"
    processed_files = set()
    if checkpoint_path.exists():
        try:
            with open(checkpoint_path, "r") as f:
                checkpoint = json.load(f)
                all_results = checkpoint.get("results", [])
                summary_rows = checkpoint.get("summary", [])
                processed_files = {r["filename"] for r in all_results}
            print(f"  ► Resuming from checkpoint: {len(processed_files)}/{len(video_files)} already done")
        except Exception as e:
            print(f"  ► Checkpoint corrupt, starting fresh: {e}")

    for i, vf in enumerate(video_files):
        print(f"\n[{i+1}/{len(video_files)}] {vf.name}")

        if vf.name in processed_files:
            print(f"    Skipping (already in checkpoint)")
            continue

        # Run analysis
        try:
            result = analyze_single_video(vf, temp_dir)
        except Exception as e:
            print(f"    ERROR: Skipping {vf.name} — {e}")
            continue

        # Match metadata
        video_id = None
        for vid in metadata:
            if vid in vf.stem:
                video_id = vid
                result["metadata"] = metadata[vid]
                break

        all_results.append(result)

        # Build summary row
        summary = {
            "filename": vf.name,
            "video_id": video_id or "",
            "channel": result.get("metadata", {}).get("channel_name", ""),
            "viral_status": result.get("metadata", {}).get("viral_status", ""),
            "domain": result.get("metadata", {}).get("domain", ""),
            "lean": result.get("metadata", {}).get("lean", ""),
            "duration_seconds": result.get("cuts", {}).get("duration_seconds", 0),
            "cuts_per_minute": result.get("cuts", {}).get("cuts_per_minute", 0),
            "full_audio_seconds": result.get("denominators", {}).get("full_audio_seconds", 0),
            "speech_covered_seconds": result.get("denominators", {}).get("speech_covered_seconds", 0),
            "non_speech_seconds": result.get("denominators", {}).get("non_speech_seconds", 0),
            "speech_density_pct": result.get("denominators", {}).get("speech_density_pct", 0),
            "silence_pct": result.get("silence", {}).get("silence_percentage", 0),
            "wpm": result.get("speech", {}).get("wpm", 0),
            "music_detected": result.get("music", {}).get("music_detected", False),
            "tempo_bpm": result.get("music", {}).get("estimated_tempo_bpm", 0),
            "high_activation_pct": result.get("manipulation_metrics", {}).get("high_activation_pct", 0),
            "recovery_pct": result.get("manipulation_metrics", {}).get("recovery_pct", 0),
            "recovery_periods": result.get("manipulation_metrics", {}).get("recovery_period_count", 0),
            "max_activation_streak": result.get("manipulation_metrics", {}).get("max_activation_streak", 0),
            "energy_cv": result.get("manipulation_metrics", {}).get("energy_cv", 0),
            "pitch_cv": result.get("manipulation_metrics", {}).get("pitch_cv", 0),
            "relentlessness_composite": result.get("manipulation_metrics", {}).get("relentlessness_composite", 0),
            "activation_pressure": result.get("manipulation_metrics", {}).get("activation_pressure", 0),
            "recovery_scarcity": result.get("manipulation_metrics", {}).get("recovery_scarcity", 0),
            "space_compression": result.get("manipulation_metrics", {}).get("space_compression", 0),
        }
        summary_rows.append(summary)

        # Print quick summary
        print(f"    Cuts/min: {summary['cuts_per_minute']} | "
              f"Silence: {summary['silence_pct']}% | "
              f"WPM: {summary['wpm']} | "
              f"High activation: {summary['high_activation_pct']}% | "
              f"Recovery: {summary['recovery_periods']} periods | "
              f"Relentlessness: {summary['relentlessness_composite']}")

        # Save checkpoint after each video (Colab crash protection)
        try:
            with open(checkpoint_path, "w") as f:
                json.dump({"results": all_results, "summary": summary_rows}, f, default=str)
        except Exception:
            pass  # Don't crash the pipeline over a checkpoint write failure

    # ── SAVE OUTPUTS ─────────────────────────────────────────

    # 1. Full results (JSON) — includes all time-series
    full_json = output_dir / "full_analysis.json"
    with open(full_json, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nFull analysis saved to {full_json}")

    # 2. Summary CSV — one row per video with key metrics
    summary_csv = output_dir / "analysis_summary.csv"
    if summary_rows:
        with open(summary_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
            writer.writeheader()
            writer.writerows(summary_rows)
    print(f"Summary CSV saved to {summary_csv}")

    # 3. Paired comparison (viral vs average within creator)
    paired_csv = output_dir / "paired_comparison.csv"
    pairs = build_paired_comparison(summary_rows)
    if pairs:
        with open(paired_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=pairs[0].keys())
            writer.writeheader()
            writer.writerows(pairs)
    print(f"Paired comparison saved to {paired_csv}")

    # Cleanup temp dir
    if temp_dir.exists():
        import shutil
        shutil.rmtree(temp_dir)

    # Print final summary
    print(f"\n{'='*60}")
    print(f" ANALYSIS COMPLETE")
    print(f"{'='*60}")
    print(f" Videos analyzed: {len(all_results)}")

    if summary_rows:
        viral = [s for s in summary_rows if s["viral_status"] == "viral"]
        average = [s for s in summary_rows if s["viral_status"] == "average"]

        if viral and average:
            print(f"\n VIRAL vs AVERAGE (means):")
            for metric in ["cuts_per_minute", "silence_pct", "wpm",
                           "high_activation_pct", "recovery_periods",
                           "energy_cv", "relentlessness_composite"]:
                v_mean = np.mean([s[metric] for s in viral])
                a_mean = np.mean([s[metric] for s in average])
                print(f"   {metric:30s}  viral={v_mean:8.2f}  avg={a_mean:8.2f}  diff={v_mean-a_mean:+8.2f}")

    print(f"\n Files:")
    print(f"   {full_json}")
    print(f"   {summary_csv}")
    print(f"   {paired_csv}")
    print(f"{'='*60}\n")


def build_paired_comparison(summary_rows):
    """
    Build within-creator viral vs average comparisons.
    For each creator, compute mean metrics for viral and average,
    then compute the difference.
    """
    from collections import defaultdict

    by_creator = defaultdict(lambda: {"viral": [], "average": []})
    for row in summary_rows:
        channel = row["channel"]
        status = row["viral_status"]
        if channel and status in ("viral", "average"):
            by_creator[channel][status].append(row)

    metrics_to_compare = [
        "cuts_per_minute", "silence_pct", "wpm",
        "high_activation_pct", "recovery_pct", "recovery_periods",
        "max_activation_streak", "energy_cv", "pitch_cv",
        "relentlessness_composite",
    ]

    pairs = []
    for channel, groups in by_creator.items():
        if not groups["viral"] or not groups["average"]:
            continue

        pair = {
            "channel": channel,
            "domain": groups["viral"][0]["domain"],
            "lean": groups["viral"][0]["lean"],
            "viral_count": len(groups["viral"]),
            "average_count": len(groups["average"]),
        }

        for metric in metrics_to_compare:
            v_vals = [s[metric] for s in groups["viral"] if isinstance(s[metric], (int, float))]
            a_vals = [s[metric] for s in groups["average"] if isinstance(s[metric], (int, float))]

            v_mean = np.mean(v_vals) if v_vals else 0
            a_mean = np.mean(a_vals) if a_vals else 0

            pair[f"viral_{metric}"] = round(v_mean, 2)
            pair[f"average_{metric}"] = round(a_mean, 2)
            pair[f"diff_{metric}"] = round(v_mean - a_mean, 2)

        pairs.append(pair)

    return pairs


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Structural Analysis Pipeline")
    parser.add_argument("--input", "-i", default="./data/videos/",
                        help="Directory containing downloaded video files")
    parser.add_argument("--metadata", "-m", default="./data/collected_videos.csv",
                        help="Path to collected_videos.csv from Step 1")
    parser.add_argument("--output", "-o", default="./data/analysis/",
                        help="Output directory for analysis results")
    args = parser.parse_args()

    run_pipeline(args.input, args.metadata, args.output)
