# vsearch - Decision Model Video Search

A lightweight command-line interface that searches local video files using an Ollama decision model as the sole judge. No CLIP embeddings or pre-built `.npz` caches are required — frames are extracted and validated on the fly.

## Quick Start

1. Install dependencies:

   ```
   pip install -r requirements.txt
   ```

2. Search a video with a text prompt:

   ```
   vsearch search --video myvideo.mp4 --ollama-state "A user is searching for: a person running" --ollama-instructions "Does the image match the query: a person running?" -k 5
   ```

3. Limit the number of results:

   ```
   vsearch search --video myvideo.mp4 --ollama-state "A user is searching for: a person running" --ollama-instructions "Does the image match the query: a person running?" -k 5
   ```

4. Save matched frames as JPEG images:

   ```
   vsearch search --video myvideo.mp4 --ollama-state "A user is searching for: a person running" --ollama-instructions "Does the image match the query: a person running?" -sf /tmp/frames
   ```

5. Output JSON:

   ```
   vsearch search --video myvideo.mp4 --ollama-state "A user is searching for: a person running" --ollama-instructions "Does the image match the query: a person running?" --json
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
| `--ollama-state` | `-s` | State context string sent to Ollama (default: empty) |
| `--ollama-instructions` | `-oi` | Instructions string sent to Ollama (default: empty) |
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

## Configuration (`config.toml`)

The config file contains a single `[search]` section:

```toml
[search]
top_k = 5
json_output = false
save_frames = ""
video = ""
decision_model_floor = 0.8
decision_model_sample_rate = 3
ollama_host = "192.168.1.190"
ollama_frame_scale = 0.2
ollama_decision_model = ["clef-flash:9b", "d3:4b"]
ollama_workers = 4
scene_change = false
scene_threshold = 0.4
ollama_state_template = ""
ollama_instructions_template = ""

### Key config values

- `top_k` — maximum results returned (default: `5`)
- `json_output` — default to JSON output (`true`/`false`)
- `save_frames` — default output directory for saved frames
- `decision_model_floor` — Ollama score threshold (default: `0.8`)
- `decision_model_sample_rate` — sampling interval in seconds (default: `3`)
- `ollama_host` — Ollama endpoint (falls back to `localhost` if unreachable)
- `ollama_frame_scale` — image scale factor sent to Ollama (default: `0.2`)
- `ollama_decision_model` — model name or fallback list (default: `["clef-flash:9b", "d3:4b"]`)
- `ollama_workers` — concurrent validation threads (default: `4`)
- `scene_change` — use scene-change detection instead of blind sampling (default: `false`)
- `scene_threshold` — scene detection sensitivity (default: `0.4`)
- `ollama_state_template` — state context string sent to Ollama (default: empty)
- `ollama_instructions_template` — instructions string sent to Ollama (default: empty)

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
- `query_hash` — MD5 of the state/instructions query
- `params_hash` — MD5 of `(decision_model_floor, decision_model_sample_rate, ollama_frame_scale, scene_change, scene_threshold, ollama_state, ollama_instructions)`

### Lookup Cache (`--cache`)

Use `--cache /path/to/cache` to look up existing cached results. If a matching cache file exists (same video, query, and parameters), the search is skipped and cached results are returned instantly:

```bash
# First run: no cache, performs full search
vsearch search --video myvideo.mp4 --ollama-state "A user is searching for: a person" --ollama-instructions "Does the image match the query: a person?" --save-cache /tmp/cache

# Second run: cache hit, returns instantly without Ollama
vsearch search --video myvideo.mp4 --ollama-state "A user is searching for: a person" --ollama-instructions "Does the image match the query: a person?" --cache /tmp/cache
```

### Save Cache (`--save-cache`)

Use `--save-cache /path/to/cache` (or just `--save-cache` to default to the current directory) to save results after a search completes. The file is written only if the search actually ran (not when a cache hit occurred):

```bash
# Save to explicit directory
vsearch search --video myvideo.mp4 --ollama-state "A user is searching for: a person" --ollama-instructions "Does the image match the query: a person?" --save-cache /tmp/cache

# Save to current directory (.)
vsearch search --video myvideo.mp4 --ollama-state "A user is searching for: a person" --ollama-instructions "Does the image match the query: a person?" --save-cache

# Combined: look up cache first, save new results if not found
vsearch search --video myvideo.mp4 --ollama-state "A user is searching for: a person" --ollama-instructions "Does the image match the query: a person?" --cache /tmp/cache --save-cache /tmp/cache
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
  "query": "A user is searching for: a person running",
  "parameters": {
    "decision_model_floor": 0.8,
    "decision_model_sample_rate": 3,
    "ollama_frame_scale": 0.2,
    "scene_change": false,
    "scene_threshold": 0.4,
    "ollama_state_template": "A user is searching for: {query}",
    "ollama_instructions_template": "Does the image match the query: {query}?"
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
vsearch search --scene-change --scene-threshold 0.5 --video video.mp4 --ollama-state "A user is searching for: a person" --ollama-instructions "Does the image match the query: a person?"
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

**Configuration:**

- `scene_change = false` – disable scene detection (uniform sampling).
- `scene_threshold = 0.4` – scene-change sensitivity threshold.

- No indexing step — videos are read directly with OpenCV.
- No `.npz` cache — frames are extracted and validated on the fly.
- Ollama-only scoring — `final_score = Ollama_score`; CLIP is never invoked.
- Endpoint fallback — if the configured Ollama host is unreachable, `localhost:11434` is used automatically.
- Parallel validation — frames are validated concurrently via `ThreadPoolExecutor`.

