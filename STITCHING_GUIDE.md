# Stitching Guide — Merge Search-Result Timestamps into One Video

## Overview

Turn a list of timestamps (from `scs` or any source) into a single video by extracting a fixed-length clip (e.g. 3 seconds) from each source video and concatenating them in order.

---

## 1. Prerequisites

| Tool | Purpose | Verify |
|------|---------|--------|
| **ffmpeg** | Extract clips, normalize codecs, concatenate | `ffmpeg -version` |
| **ffprobe** | Inspect codec/resolution/FPS | `ffprobe -version` |
| **jq** (optional) | Parse JSON results | `jq --version` |
| **bash** | Run scripts | `bash --version` |

Install on Ubuntu/Debian:

```bash
sudo apt-get update && sudo apt-get install -y ffmpeg jq
```

---

## 2. Get Timestamps from `scs`

Example: search for an image across all caches and emit JSON:

```bash
python3 -m scs search ./cache/ \
  -i ./fffs003_spycam_01:34:20_0.2847.jpg \
  -k 2 --json > /tmp/search_results.json
```

Each result looks like:

```json
{
  "timestamp_seconds": 5660.121133333333,
  "timestamp_formatted": "01:34:20",
  "source_cache": "fffs003_spycam_clip-vit-large-patch14_3s_0821d1d0.npz",
  "source_video_path": "/mnt/nas/zpool2/jav/Sena-Asami/fffs003_spycam.mp4",
  "score": 0.9885
}
```

Extract timestamps and source videos (one per line):

```bash
jq -r '.results[] | "\(.timestamp_seconds)|\(.source_video_path)"' \
  /tmp/search_results.json > /tmp/timestamps_and_videos.txt
```

---

## 3. Workflow Summary

1. **Extract** a fixed-length clip (e.g. 3s) from each source video at the timestamp.  
2. **Normalize** every clip to the same codec / resolution / frame-rate so they concatenate cleanly.  
3. **Build** a concat list file.  
4. **Concatenate** with ffmpeg's `-f concat` demuxer.  
5. **Move** the final video to the target location.

---

## 4. Step-by-Step Commands

### 4.1 Create workspace

```bash
TMPDIR=$(mktemp -d -p /tmp stitch_segments)
cd "$TMPDIR"
```

### 4.2 Extract 3-second clips

Run once per timestamp/video pair:

```bash
# fast seek: -ss BEFORE -i for speed
ffmpeg -y -v error -ss "$ts" -i "$src_video" -t 3 -c copy "$clip.mp4"
```

*Notes:*  
- `-ss` must come **before** `-i` for fast seeking (avoids decoding the whole file).  
- `-c copy` avoids re-encoding (fastest) but fails if source clips have mismatched codecs.

### 4.3 Normalize clips (recommended)

If clips come from different videos, re-encode them to a uniform format:

```bash
ffmpeg -y -v error -i "$clip.mp4" \
  -vf "scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2,fps=30" \
  -c:v libx264 -preset fast -crf 23 \
  -c:a aac -b:a 128k -ar 44100 -ac 2 \
  "norm_$clip.mp4"
```

| Flag | Meaning |
|------|---------|
| `scale=1920:1080:force_original_aspect_ratio=decrease` | Fit inside 1920×1080 preserving aspect ratio |
| `pad=1920:1080:(ow-iw)/2:(oh-ih)/2` | Pad to 1920×1080 (black bars) |
| `fps=30` | Constant frame rate (prevents sync drift) |
| `-c:v libx264 -crf 23` | H.264 video, high quality |
| `-c:a aac -b:a 128k -ar 44100 -ac 2` | AAC audio, stereo |

### 4.4 Build concat list

```bash
for f in norm_*.mp4; do
    echo "file '$TMPDIR/$f'" >> list.txt
done
```

### 4.5 Concatenate

**Fast copy (no re-encode)** — only works if all clips share the same codec/parameters:

```bash
ffmpeg -y -v error -f concat -safe 0 -i list.txt -c copy final.mp4
```

**If copy fails** (codec mismatch), re-encode while concatenating:

```bash
ffmpeg -y -v error -f concat -safe 0 -i list.txt \
  -c:v libx264 -preset fast -crf 23 \
  -c:a aac -b:a 128k final.mp4
```

### 4.6 Move final video

```bash
mkdir -p /home/btt/repos/shot-scanner/st
mv "$TMPDIR/final.mp4" /home/btt/repos/shot-scanner/st.mp4
```

Verify:

```bash
ffprobe -v error -show_entries format=duration,size -of default=noprint_wrappers=1 /home/btt/repos/shot-scanner/st.mp4
```

---

## 5. Complete Bash Script

Save as `stitch.sh`, make executable (`chmod +x stitch.sh`), and run:

```bash
#!/usr/bin/env bash
set -euo pipefail

# --- Inputs ---
JSON="/tmp/search_results.json"
OUT="/home/btt/repos/shot-scanner/st.mp4"
SEGMENT_DURATION=3   # seconds

# --- Temp workspace ---
TMPDIR=$(mktemp -d -p /tmp stitch_segments)
cd "$TMPDIR"

# --- Extract timestamps from JSON ---
TIMESTAMPS=$(jq -r '.results[].timestamp_seconds' "$JSON")

i=0
for ts in $TIMESTAMPS; do
    src_video=$(jq -r ".results[$i].source_video_path" "$JSON")
    clip="clip_$i.mp4"
    norm="norm_$clip"

    # Extract fixed-length clip (fast seek)
    ffmpeg -y -v error -ss "$ts" -i "$src_video" -t "$SEGMENT_DURATION" -c copy "$clip"

    # Normalize to common format
    ffmpeg -y -v error -i "$clip" \
        -vf "scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2,fps=30" \
        -c:v libx264 -preset fast -crf 23 \
        -c:a aac -b:a 128k -ar 44100 -ac 2 \
        "$norm"

    rm -f "$clip"
    ((i++))
done

# --- Concat list ---
for f in norm_*.mp4; do
    echo "file '$TMPDIR/$f'" >> list.txt
done

# --- Concatenate ---
ffmpeg -y -v error -f concat -safe 0 -i list.txt -c copy final.mp4

# --- Move ---
mkdir -p "$(dirname "$OUT")"
mv final.mp4 "$OUT"

echo "Done: $OUT"
echo "Duration: $(ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1 "$OUT")"
```

---

## 6. Common Pitfalls & Fixes

| Symptom | Cause | Fix |
|---------|-------|-----|
| `[swscaler] Cannot convert interlaced to progressive frames` | Source video is interlaced (`yuv420p(tv, bt709, top first)`); OpenCV fallback fails. | Use **ffmpeg only** (`-ss` before `-i`). Never use OpenCV for interlaced extraction. |
| `Codec parameters do not match` during concat | Clips have different H.264 profiles, audio codecs, resolutions, or frame rates. | Re-encode every clip to a uniform format (see §4.3). |
| Audio out of sync | Different audio sample rates between source clips. | Normalize audio: `-c:a aac -b:a 128k -ar 44100 -ac 2` |
| Video plays but lags/stuck | Concatenation with `-c copy` on mismatched streams. | Re-encode the final output with `libx264` and `aac`. |
| ffmpeg hangs | `-ss` placed **after** `-i` causes full-file decode to the timestamp. | Put `-ss` **before** `-i` (fast seek). |
| Output file missing | `mv` failed because destination directory doesn't exist. | `mkdir -p /target/dir` before moving. |

---

## 7. Example Output

After running the full pipeline on a search with 20 results:

```
/home/btt/repos/shot-scanner/st.mp4
  Size:   32 MB
  Duration: ~61 seconds  (20 clips × 3s)
  Codec:  h264 / aac
  Resolution: 1920×1080
  FPS: 30
```

Each clip is ordered by search relevance score (highest first).
