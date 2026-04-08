"""
Video Collection Script — Production Study (500 videos, 30 channels)
=====================================================================
Collects YouTube Shorts metadata via the YouTube Data API.
For each channel, selects the top viral and average-performing videos.

Channel design:
  - 8 political right
  - 8 political left
  - 2 center / nonpartisan
  - 12 non-political

Mirrored pairing design:
  Right solo commentator  ↔  Left solo commentator
  Right institutional news ↔  Left institutional news
  Right instructional      ↔  Left instructional
  Right populist/grassroot ↔  Left populist/grassroot

Usage:
    export YOUTUBE_API_KEY="your_key_here"
    python3 collect_videos.py --output ./data/

Requires:
    pip install google-api-python-client
"""

import os
import sys
import json
import csv
import argparse
from datetime import datetime
from pathlib import Path

# ─────────────────────────────────────────────────────────────
# CHANNEL LIST — 30 CHANNELS, MIRRORED PAIRING DESIGN
# ─────────────────────────────────────────────────────────────

CHANNELS = [
    # ── POLITICAL RIGHT (8) ──────────────────────────────────

    # Solo commentators (right) — paired with left solo commentators
    {"name": "Ben Shapiro",       "handle": "@BenShapiro",        "domain": "political", "lean": "right", "format": "solo_commentator"},
    {"name": "Matt Walsh",        "handle": "@MattWalsh",         "domain": "political", "lean": "right", "format": "solo_commentator"},

    # Institutional news (right) — paired with left institutional
    {"name": "Fox News",          "handle": "@FoxNews",           "domain": "political", "lean": "right", "format": "institutional_news"},
    {"name": "Daily Wire",        "handle": "@DailyWire",         "domain": "political", "lean": "right", "format": "institutional_news"},

    # Instructional/educational (right) — paired with left instructional
    {"name": "PragerU",           "handle": "@PragerU",           "domain": "political", "lean": "right", "format": "instructional"},
    {"name": "Turning Point USA", "handle": "@TPUSA",             "domain": "political", "lean": "right", "format": "instructional"},

    # Populist/grassroots (right) — paired with left populist
    {"name": "Steven Crowder",    "handle": "@StevenCrowder",     "domain": "political", "lean": "right", "format": "populist"},
    {"name": "Charlie Kirk",      "handle": "@CharlieKirk",       "domain": "political", "lean": "right", "format": "populist"},

    # ── POLITICAL LEFT (8) ───────────────────────────────────

    # Solo commentators (left)
    {"name": "Hasan Piker",       "handle": "@HasanAbi",          "domain": "political", "lean": "left", "format": "solo_commentator"},
    {"name": "Brian Tyler Cohen", "handle": "@briantylercohen",   "domain": "political", "lean": "left", "format": "solo_commentator"},

    # Institutional news (left)
    {"name": "MSNBC",             "handle": "@MSNBC",             "domain": "political", "lean": "left", "format": "institutional_news"},
    {"name": "NowThis",           "handle": "@NowThis",           "domain": "political", "lean": "left", "format": "institutional_news"},

    # Instructional/educational (left)
    {"name": "Robert Reich",      "handle": "@RBReich",           "domain": "political", "lean": "left", "format": "instructional"},
    {"name": "Second Thought",    "handle": "@SecondThought",     "domain": "political", "lean": "left", "format": "instructional"},

    # Populist/grassroots (left)
    {"name": "The Young Turks",   "handle": "@TheYoungTurks",     "domain": "political", "lean": "left", "format": "populist"},
    {"name": "Secular Talk",      "handle": "@SecularTalk",       "domain": "political", "lean": "left", "format": "populist"},

    # ── CENTER / NONPARTISAN (2) ─────────────────────────────

    {"name": "Reuters",           "handle": "@Reuters",           "domain": "political", "lean": "center", "format": "institutional_news"},
    {"name": "Breaking Points",   "handle": "@BreakingPoints",    "domain": "political", "lean": "center", "format": "solo_commentator"},

    # ── NON-POLITICAL (12) ───────────────────────────────────

    # Productivity/self-help
    {"name": "Ali Abdaal",        "handle": "@aliabdaal",         "domain": "non-political", "lean": "", "format": "productivity"},
    {"name": "Mel Robbins",       "handle": "@melrobbins",        "domain": "non-political", "lean": "", "format": "self_help"},

    # Entertainment/spectacle
    {"name": "MrBeast",           "handle": "@MrBeast",           "domain": "non-political", "lean": "", "format": "entertainment"},
    {"name": "Dude Perfect",      "handle": "@DudePerfect",       "domain": "non-political", "lean": "", "format": "entertainment"},

    # Science/education
    {"name": "Veritasium",        "handle": "@veritasium",        "domain": "non-political", "lean": "", "format": "science"},
    {"name": "Mark Rober",        "handle": "@MarkRober",         "domain": "non-political", "lean": "", "format": "science"},

    # Fitness/health
    {"name": "Jeff Nippard",      "handle": "@JeffNippard",       "domain": "non-political", "lean": "", "format": "fitness"},
    {"name": "Athlean-X",         "handle": "@athleanx",           "domain": "non-political", "lean": "", "format": "fitness"},

    # Finance/business
    {"name": "Graham Stephan",    "handle": "@GrahamStephan",     "domain": "non-political", "lean": "", "format": "finance"},
    {"name": "Andrei Jikh",       "handle": "@AndreiJikh",        "domain": "non-political", "lean": "", "format": "finance"},

    # Cooking/lifestyle
    {"name": "Joshua Weissman",   "handle": "@JoshuaWeissman",    "domain": "non-political", "lean": "", "format": "cooking"},
    {"name": "Nick DiGiovanni",   "handle": "@NickDiGiovanni",    "domain": "non-political", "lean": "", "format": "cooking"},
]

# ─────────────────────────────────────────────────────────────
# COLLECTION CONFIG
# ─────────────────────────────────────────────────────────────

VIRAL_COUNT = 5        # Top N viral Shorts per channel (by view count)
AVERAGE_COUNT = 5      # N average-performing Shorts per channel
# Selection is within-channel and relative: viral = top-N by view count,
# average = sampled from the middle of that channel's own distribution.
# No fixed absolute view-count thresholds are used.
# Targeting ~300 videos (30 channels × 10 each), with some channels
# potentially having fewer Shorts available. Overflow channels provide buffer.

MIN_DURATION = 15      # Minimum Short duration (seconds)
MAX_DURATION = 61      # Maximum Short duration (seconds)


# ─────────────────────────────────────────────────────────────
# YOUTUBE API HELPERS
# ─────────────────────────────────────────────────────────────

def get_youtube_client():
    """Initialize YouTube Data API client."""
    from googleapiclient.discovery import build

    api_key = os.environ.get("YOUTUBE_API_KEY")
    if not api_key:
        print("ERROR: Set YOUTUBE_API_KEY environment variable")
        print("  export YOUTUBE_API_KEY='your_key_here'")
        sys.exit(1)

    return build("youtube", "v3", developerKey=api_key)


def get_channel_id(youtube, handle):
    """Resolve a @handle to a channel ID."""
    # Try exact handle lookup first
    clean_handle = handle.lstrip("@")
    request = youtube.channels().list(
        part="id",
        forHandle=clean_handle,
    )
    response = request.execute()
    if response.get("items"):
        return response["items"][0]["id"]

    # Fallback: fuzzy search
    request = youtube.search().list(
        part="snippet",
        q=handle,
        type="channel",
        maxResults=1,
    )
    response = request.execute()

    if response.get("items"):
        return response["items"][0]["snippet"]["channelId"]

    return None


def get_channel_shorts(youtube, channel_id, max_results=200):
    """
    Fetch Shorts from a channel.
    YouTube doesn't have a direct "Shorts" filter in the API,
    so we fetch short-duration videos and filter by duration.
    """
    all_videos = []
    next_page = None

    while len(all_videos) < max_results:
        request = youtube.search().list(
            part="id,snippet",
            channelId=channel_id,
            type="video",
            order="viewCount",
            maxResults=50,
            videoDuration="short",  # Under 4 minutes
            pageToken=next_page,
        )
        response = request.execute()

        video_ids = [item["id"]["videoId"] for item in response.get("items", [])]

        if not video_ids:
            break

        # Get full video details (duration, view count, etc.)
        details_request = youtube.videos().list(
            part="contentDetails,statistics,snippet",
            id=",".join(video_ids),
        )
        details = details_request.execute()

        for item in details.get("items", []):
            duration_str = item["contentDetails"]["duration"]
            duration_sec = parse_duration(duration_str)

            # Filter to actual Shorts (15-60 seconds)
            if MIN_DURATION <= duration_sec <= MAX_DURATION:
                views = int(item["statistics"].get("viewCount", 0))
                all_videos.append({
                    "video_id": item["id"],
                    "title": item["snippet"]["title"],
                    "published_at": item["snippet"]["publishedAt"],
                    "duration_seconds": duration_sec,
                    "view_count": views,
                    "like_count": int(item["statistics"].get("likeCount", 0)),
                    "comment_count": int(item["statistics"].get("commentCount", 0)),
                    "url": f"https://www.youtube.com/shorts/{item['id']}",
                })

        next_page = response.get("nextPageToken")
        if not next_page:
            break

    return all_videos


def parse_duration(duration_str):
    """Parse ISO 8601 duration (PT1M30S) to seconds."""
    import re
    match = re.match(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", duration_str)
    if not match:
        return 0
    hours = int(match.group(1) or 0)
    minutes = int(match.group(2) or 0)
    seconds = int(match.group(3) or 0)
    return hours * 3600 + minutes * 60 + seconds


def select_viral_and_average(videos, viral_count=5, average_count=5):
    """
    Select viral (top N by views) and average (median range) videos.
    Returns two lists: viral_videos, average_videos.
    """
    if not videos:
        return [], []

    sorted_vids = sorted(videos, key=lambda v: v["view_count"], reverse=True)

    # Viral: top N
    viral = sorted_vids[:viral_count]

    # Average: from the middle of the distribution
    # Skip top viral and bottom (very low/failed), take from median band
    remaining = sorted_vids[viral_count:]
    if len(remaining) < average_count:
        average = remaining
    else:
        mid_start = len(remaining) // 4
        mid_end = mid_start + average_count * 2
        mid_pool = remaining[mid_start:mid_end]
        # Sample evenly from the middle
        if len(mid_pool) >= average_count:
            step = max(1, len(mid_pool) // average_count)
            average = mid_pool[::step][:average_count]
        else:
            average = mid_pool[:average_count]

    return viral, average


# ─────────────────────────────────────────────────────────────
# MAIN COLLECTION PIPELINE
# ─────────────────────────────────────────────────────────────

def run_collection(output_dir):
    """Run the full collection pipeline."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    youtube = get_youtube_client()

    all_selected = []
    channel_summary = []

    print(f"\n{'='*70}")
    print(f" VIDEO COLLECTION — PRODUCTION STUDY")
    print(f" Channels: {len(CHANNELS)}")
    print(f" Target: {VIRAL_COUNT} viral + {AVERAGE_COUNT} average per channel")
    print(f" Expected total: ~{len(CHANNELS) * (VIRAL_COUNT + AVERAGE_COUNT)} videos")
    print(f"{'='*70}\n")

    for i, ch in enumerate(CHANNELS):
        print(f"\n[{i+1}/{len(CHANNELS)}] {ch['name']} ({ch['handle']}) — {ch['lean'] or 'non-political'}")

        # Resolve channel ID
        channel_id = get_channel_id(youtube, ch["handle"])
        if not channel_id:
            print(f"    WARNING: Could not resolve channel ID for {ch['handle']}")
            channel_summary.append({
                "name": ch["name"], "handle": ch["handle"],
                "domain": ch["domain"], "lean": ch["lean"],
                "format": ch["format"],
                "total_shorts_found": 0, "selected": 0,
                "status": "CHANNEL_NOT_FOUND",
            })
            continue

        # Fetch Shorts
        print(f"    Fetching Shorts...")
        shorts = get_channel_shorts(youtube, channel_id)
        print(f"    Found {len(shorts)} Shorts")

        if not shorts:
            channel_summary.append({
                "name": ch["name"], "handle": ch["handle"],
                "domain": ch["domain"], "lean": ch["lean"],
                "format": ch["format"],
                "total_shorts_found": 0, "selected": 0,
                "status": "NO_SHORTS_FOUND",
            })
            continue

        # Select viral and average
        viral, average = select_viral_and_average(shorts, VIRAL_COUNT, AVERAGE_COUNT)

        # Tag and collect
        for v in viral:
            v["channel_name"] = ch["name"]
            v["channel_handle"] = ch["handle"]
            v["domain"] = ch["domain"]
            v["lean"] = ch["lean"]
            v["format_type"] = ch["format"]
            v["viral_status"] = "viral"
            all_selected.append(v)

        for a in average:
            a["channel_name"] = ch["name"]
            a["channel_handle"] = ch["handle"]
            a["domain"] = ch["domain"]
            a["lean"] = ch["lean"]
            a["format_type"] = ch["format"]
            a["viral_status"] = "average"
            all_selected.append(a)

        selected = len(viral) + len(average)
        print(f"    Selected: {len(viral)} viral, {len(average)} average")
        if viral:
            print(f"    Viral view range: {viral[-1]['view_count']:,} — {viral[0]['view_count']:,}")
        if average:
            print(f"    Average view range: {average[-1]['view_count']:,} — {average[0]['view_count']:,}")

        channel_summary.append({
            "name": ch["name"], "handle": ch["handle"],
            "domain": ch["domain"], "lean": ch["lean"],
            "format": ch["format"],
            "total_shorts_found": len(shorts),
            "selected": selected,
            "viral_selected": len(viral),
            "average_selected": len(average),
            "status": "OK",
        })

    # ── SAVE OUTPUTS ─────────────────────────────────────────

    # 1. Full JSON
    json_path = output_dir / "collected_videos.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_selected, f, indent=2, ensure_ascii=False)
    print(f"\nJSON saved to {json_path}")

    # 2. CSV (for pipeline compatibility)
    csv_path = output_dir / "collected_videos.csv"
    if all_selected:
        fieldnames = list(all_selected[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_selected)
    print(f"CSV saved to {csv_path}")

    # 3. Channel summary
    summary_path = output_dir / "channel_summary.csv"
    if channel_summary:
        with open(summary_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=channel_summary[0].keys())
            writer.writeheader()
            writer.writerows(channel_summary)
    print(f"Channel summary saved to {summary_path}")

    # 4. Generate download script
    script_path = output_dir / "download_videos.sh"
    generate_download_script(all_selected, script_path)
    print(f"Download script saved to {script_path}")

    # ── SUMMARY ──────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f" COLLECTION COMPLETE")
    print(f"{'='*70}")
    print(f" Total videos selected: {len(all_selected)}")
    print(f" Viral: {sum(1 for v in all_selected if v['viral_status'] == 'viral')}")
    print(f" Average: {sum(1 for v in all_selected if v['viral_status'] == 'average')}")
    print(f"\n By lean:")
    for lean in ["right", "left", "center", ""]:
        label = lean if lean else "non-political"
        count = sum(1 for v in all_selected if v["lean"] == lean)
        print(f"   {label}: {count}")
    print(f"\n Next steps:")
    print(f"   1. Review collected_videos.csv")
    print(f"   2. Run: bash {script_path} --cookies-from-browser firefox")
    print(f"   3. Run: python3 analyze_videos.py --input ./data/videos/ --metadata {csv_path}")
    print(f"{'='*70}\n")


def generate_download_script(videos, output_path):
    """Generate a bash script to download all selected videos with yt-dlp."""
    lines = [
        "#!/bin/bash",
        "# Auto-generated download script — Production Study",
        f"# Generated: {datetime.now().isoformat()}",
        f"# Total videos: {len(videos)}",
        "# Requires: pip install yt-dlp",
        "",
        '# Pass --cookies-from-browser firefox (or chrome) as first argument',
        'COOKIES_ARG="${1:---cookies-from-browser firefox}"',
        "",
        'OUTDIR="./data/videos"',
        'mkdir -p "$OUTDIR"',
        "",
        'TOTAL=' + str(len(videos)),
        'COUNT=0',
        'FAILED=0',
        "",
    ]

    for v in videos:
        safe_channel = v["channel_name"].replace(" ", "_").replace("'", "")
        filename = f"{safe_channel}_{v['viral_status']}_{v['video_id']}"
        title_preview = v["title"][:50].replace('"', '\\"')

        lines.append(f'COUNT=$((COUNT + 1))')
        lines.append(f'echo "[$COUNT/$TOTAL] {title_preview}..."')
        lines.append(
            f'yt-dlp -o "$OUTDIR/{filename}.%(ext)s" '
            f'"{v["url"]}" '
            f'--quiet $COOKIES_ARG || FAILED=$((FAILED + 1))'
        )
        lines.append('sleep 1')
        lines.append('')

    lines.append('echo ""')
    lines.append('echo "Download complete: $((TOTAL - FAILED))/$TOTAL succeeded, $FAILED failed"')
    lines.append('')

    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    os.chmod(output_path, 0o755)


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect YouTube Shorts for production study")
    parser.add_argument("--output", "-o", default="./data/",
                        help="Output directory for collected metadata")
    args = parser.parse_args()

    run_collection(args.output)
