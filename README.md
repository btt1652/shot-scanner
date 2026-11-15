# vsearch - Decision Model Video Search

A lightweight command-line interface that searches local video files using an Ollama decision model as the sole judge. No CLIP embeddings or pre-built `.npz` caches are required — frames are extracted and validated on the fly.

## Quick Start

1. Install dependencies:

   ```
   pip install -r requirements.txt
   ```

2. Search a video with a text prompt:

   ```
   vsearch search --video myvideo.mp4 -t "a person running"
   ```

3. Limit the number of results:

   ```
   vsearch search --video myvideo.mp4 -t "a person running" -k 5
   ```

4. Save matched frames as JPEG images:

   ```
   vsearch search --video myvideo.mp4 -t "a person running" -sf /tmp/frames
   ```

5. Output JSON:

   ```
   vsearch search --video myvideo.mp4 -t "a person running" --json
   ```

## Decision-Model-Only Mode

`vsearch search` uses the Ollama decision model as the sole judge. It does **not** use CLIP embeddings, `.npz` caches, or fused scores.

### Workflow

1. Frames are sampled from the video at `--decision-model-sample-rate` intervals.
2. Each frame is resized (`--ollama-frame-scale`) and sent to the Ollama server.
3. Frames whose score is below `--decision-model-floor` are discarded.
4. Passing frames are sorted by Ollama score descending.
5. Top-K / Global-K caps are applied.

### Scoring

```
final_score = Ollama_score
```

CLIP is not involved in scoring at all.

## CLI Flags

| Flag | Short | Description |
|------|-------|-------------|
| `--video` | `-vid` | Path to a video file, directory, or glob pattern (required) |
| `--text` | `-t` | Natural-language search prompt |
| `--image` | `-i` | Path to a reference image file (used as the query string) |
| `--top-k` | `-k` | Maximum number of results to return (default: all passing) |
| `--global-k` | `-gk` | Global top-K across all videos, overriding `--top-k` |
| `--json` | | Output results as strict JSON |
| `--save-frames` | `-sf` | Directory to save matched frame JPEGs |
| `--decision-model-floor` | `-dmt` | Ollama score threshold (default: 0.5) |
| `--decision-model-sample-rate` | `-dmsr` | Frame sampling interval in seconds (default: 1) |
| `--ollama-workers` | `-ow` | Concurrent Ollama validation threads (default: 1) |
| `--ollama-host` | `-oh` | Ollama server address (default: `192.168.1.190`). Falls back to `localhost` if unreachable |
| `--ollama-frame-scale` | `-ofs` | Scale factor for frames sent to Ollama (0.0–1.0, default: 0.2). Lower = faster |
| `--ollama-model` | `-dm` | Ollama model name(s), comma-separated fallback list |
| `--recursive` | `-R` | Recursively scan directories for video files |
| `--scene-change` | | Use scene-change detection instead of blind sampling |
| `--scene-threshold` | | Scene detection sensitivity, 0.0–1.0 (default: 0.4) |
| `--sample-merge` | `-sm` | Merge adjacent passing samples into ranges (blind sampling only) |
| `--merge-gap` | `-mg` | Max seconds between samples to merge (default: same as `--decision-model-sample-rate`) |

## Configuration (`config.toml`)

The config file contains a single `[search]` section:

```toml
[search]
top_k = -1
json_output = false
save_frames = ""
video = ""
decision_model_floor = 0.5
decision_model_sample_rate = 1
ollama_host = "192.168.1.190"
ollama_frame_scale = 0.2
ollama_decision_model = ["d3:9b", "d3:4b"]
ollama_workers = 1
scene_change = false
scene_threshold = 0.4
sample_merge = false
merge_gap = 60
```

### Key config values

- `top_k` — maximum results returned (default: `-1` = all passing)
- `json_output` — default to JSON output (`true`/`false`)
- `save_frames` — default output directory for saved frames
- `decision_model_floor` — Ollama score threshold (default: `0.5`)
- `decision_model_sample_rate` — sampling interval in seconds (default: `1`)
- `ollama_host` — Ollama endpoint (falls back to `localhost` if unreachable)
- `ollama_frame_scale` — image scale factor sent to Ollama (default: `0.2`)
- `ollama_decision_model` — model name or fallback list
- `ollama_workers` — concurrent validation threads (default: `1`)
- `scene_change` — use scene-change detection instead of blind sampling (default: `false`)
- `scene_threshold` — scene detection sensitivity (default: `0.4`)

## Performance Tips

- **Scale down images** with `--ollama-frame-scale` (e.g. `0.15`) — lower = faster inference.
- **Parallel validation** with `--ollama-workers 4` (or higher) — each thread handles one frame.
- **Coarser sampling** with `--decision-model-sample-rate 5` — one frame every 5 seconds.
- **Lower the threshold** with `--decision-model-floor 0.4` to keep more passing frames.
- **Scene-change detection** with `--scene-change` — automatically extracts frames at scene boundaries instead of uniform sampling, often much more efficient.

## Architecture

- **No indexing step** — videos are read directly with OpenCV.
- **No `.npz` cache** — frames are extracted and validated on the fly.
- **Ollama-only scoring** — `final_score = Ollama_score`; CLIP is never invoked.
- **Endpoint fallback** — if the configured Ollama host is unreachable, `localhost:11434` is used automatically.
- **Parallel validation** — frames are validated concurrently via `ThreadPoolExecutor`.


## Search Cache

Search results can be cached to JSON files to avoid re-running expensive Ollama validation on the same video + query combination. Cache is **opt-in** — it's only used when you explicitly pass the `--cache` or `--save-cache` flags.

### Cache File Naming

Cache files are named as:

```
<video_name>_<video_hash>_<query_hash>_<params_hash>.json
```

Where:
- `video_name` — filename without extension
- `video_hash` — SHA256 of video path + size + modification time
- `query_hash` — MD5 of the text/image query
- `params_hash` — MD5 of `(decision_model_floor, decision_model_sample_rate, ollama_frame_scale, scene_change, scene_threshold)`

### Lookup Cache (`--cache`)

Use `--cache /path/to/cache` to look up existing cached results. If a matching cache file exists (same video, query, and parameters), the search is skipped and cached results are returned instantly:

```bash
# First run: no cache, performs full search
vsearch search --video myvideo.mp4 --text "a person" --save-cache /tmp/cache

# Second run: cache hit, returns instantly without Ollama
vsearch search --video myvideo.mp4 --text "a person" --cache /tmp/cache
```

### Save Cache (`--save-cache`)

Use `--save-cache /path/to/cache` (or just `--save-cache` to default to the current directory) to save results after a search completes. The file is written only if the search actually ran (not when a cache hit occurred):

```bash
# Save to explicit directory
vsearch search --video myvideo.mp4 --text "a person" --save-cache /tmp/cache

# Save to current directory (.)
vsearch search --video myvideo.mp4 --text "a person" --save-cache

# Combined: look up cache first, save new results if not found
vsearch search --video myvideo.mp4 --text "a person" --cache /tmp/cache --save-cache /tmp/cache
```

### Cache Hit Behavior

- **Cache hit**: No Ollama calls, no frame extraction — results loaded from disk
- **Cache miss**: Full search runs, results saved to cache directory (if `--save-cache` provided)
- **Parameter mismatch**: If parameters differ (e.g., different threshold), cache is treated as miss
- **Video change**: If video file size or modification time changes, cache is invalidated

### Cache File Format

```json
{
  "video_path": "/mnt/hdd/myvideo.mp4",
  "query": "a person running",
  "parameters": {
    "decision_model_floor": 0.5,
    "decision_model_sample_rate": 3,
    "ollama_frame_scale": 0.2,
    "scene_change": false,
    "scene_threshold": 0.4
  },
  "results": [
    {
      "timestamp_seconds": 125.3,
      "timestamp_formatted": "00:02:05",
      "score": 0.6412,
      "source_video_path": "/mnt/hdd/myvideo.mp4"
    }
  ],
  "metadata": {
    "total_extracted": 45,
    "total_validated": 40,
    "search_time_seconds": 12.5,
    "cached": false,
    "created_at": "2025-01-15T10:30:00Z"
  }
}
```

## Scene Change Detection

The `--scene-change` and `--scene-threshold` flags enable automatic scene change detection. Instead of blindly sampling every `decision_model_sample_rate` seconds, the app uses FFmpeg’s scene filter to detect scenes and only validates the key frames at those transitions:

```bash
vsearch search --scene-change --scene-threshold 0.5 --video video.mp4 --text "a person"
```

**How it works:**

1. `ffmpeg` runs `-vf "select='gt(scene,THRESHOLD)',metadata=print"` on the video and parses the scene-change timestamps (`pts_time`) from stderr.
2. For each timestamp, the app extracts that frame (using OpenCV seek) and passes it to Ollama.
3. Only scene boundaries are validated, which can drastically reduce processing time and focus validation on moments that likely contain content changes.

**Performance benefits:**
- Many scenes can be detected with fewer frames than uniform sampling.
- Scene thresholds (`0.0‑1.0`) control sensitivity: `0.4` (default) captures moderate changes; `0.7` requires strong changes.
- If no scene changes are detected (e.g., static videos), the app falls back to keyframe detection.
- Scene detection is performed once and then frames are processed using the same validation pipeline as normal sampling.

**Configuration:**
- `scene_change = false` – disable scene detection (uniform sampling).
- `scene_threshold = 0.4` – scene-change sensitivity threshold.

- **No indexing step** — videos are read directly with OpenCV.
- **No `.npz` cache** — frames are extracted and validated on the fly.
- **Ollama-only scoring** — `final_score = Ollama_score`; CLIP is never invoked.
- **Endpoint fallback** — if the configured Ollama host is unreachable, `localhost:11434` is used automatically.
- **Parallel validation** — frames are validated concurrently via `ThreadPoolExecutor`.
## Sample Merge (Advanced)

If you want to merge adjacent passing samples into contiguous time ranges (useful for finding continuous action segments), use the following flags:

| Flag | Description |
|---|---|
| `--sample-merge` / `-sm` | Enable merging of adjacent passing samples into ranges (blind sampling mode only) |
| `--merge-gap SECONDS` / `-mg SECONDS` | Max seconds between adjacent samples to merge (default: same as `--decision-model-sample-rate`) |

### Example

```bash
# Merge samples within 60 seconds into ranges
vsearch search --video movie.mp4 --text "a person running" \
  --sample-merge --merge-gap 60 --top-k 5
```

### Output Format

When `--sample-merge` is enabled, the text output shows ranges instead of individual timestamps:

```
ts=00:01:23-00:02:45  score=0.72  (max=0.72, avg=0.61, n=6)
```

Where:
- `START-END`: The earliest and latest sample in the range
- `score=MAX`: Maximum score among samples in the range
- `max=MAX`: Same as score (for compatibility)
- `avg=AVG`: Average score
- `n=NUM`: Number of samples in the range

### JSON Output

The JSON output includes the full range details:
```json
{
  "start_seconds": 83.0,
  "end_seconds": 165.0,
  "start_formatted": "00:01:23",
  "end_formatted": "00:02:45",
  "duration_seconds": 82.0,
  "score_max": 0.72,
  "score_avg": 0.61,
  "score_min": 0.55,
  "sample_count": 6,
  "samples": [
    {"timestamp_seconds": 83.0, "timestamp_formatted": "00:01:23", "score": 0.72},
    {"timestamp_seconds": 105.0, "timestamp_formatted": "00:01:45", "score": 0.61}
  ]
}
```

### Cache Interaction

The cache stores raw samples (not merged ranges). Merging happens at output time, so you can:
- Re-run with different `--merge-gap` values without re-running Ollama
- Cache hit + `--sample-merge` → instantly load and re-merge

### Blind Sampling Mode Only

Sample merge only works in blind sampling mode (default). Scene-change mode is not merged since scene timestamps are not uniformly spaced.

