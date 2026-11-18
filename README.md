# vsearch — Semantic Video Frame Classification

A lightweight CLI that classifies video frames using an Ollama choice-question
model as the sole judge. No CLIP embeddings or pre-built `.npz` caches are
required — frames are extracted and validated on the fly.

## Quick Start

1. Install dependencies:

   ```
   pip install -r requirements.txt
   ```

2. Classify a video with a choice question:

   ```
   vsearch classify myvideo.mp4 -k 5
   ```

3. Customize the question and criteria (also configurable in `config.toml`):

   ```
   vsearch classify myvideo.mp4 \
     --ollama-classify-question team \
     --ollama-classify-instructions "Which team should handle this scene?" \
     --ollama-classify-criteria "billing: Payments|technical: Bugs|sales: Pricing"
   ```

4. Filter results by choice label (comma-separated for multiple):

   ```
   vsearch classify myvideo.mp4 --top-choice technical
   ```

5. Output JSON:

   ```
   vsearch classify myvideo.mp4 --json
   ```

## How It Works

1. Frames are sampled from the video at `--decision-model-sample-rate` intervals.
2. Each frame is resized (`--ollama-frame-scale`) and base64-encoded.
3. Consecutive frames with similarity ≥ `same_scene_threshold` (default 0.80) are
   grouped into same-scene batches (max 4 frames per batch, constrained by
   Ollama's multi-image limit) and sent to Ollama as a single multi-image request.
4. Ollama returns a probability distribution over the choice criteria; the
   `top_choice` and `top_probability` are extracted.
5. Frames below `--decision-model-floor` (default 0.5) are discarded.
6. Passing frames are sorted by `top_probability` descending, then capped with
   `--top-k` / `--global-k`.
7. `--top-choice` filters results by label after floor check.

## CLI Flags

| Flag | Short | Description |
|------|-------|-------------|
| `--top-k` | `-k` | Maximum number of top results to return (default: all passing) |
| `--global-k` | `-gk` | Global top-K across all videos, overriding `--top-k` |
| `--top-choice` | `-tc` | Only keep results where `top_choice` matches (comma-separated for multiple) |
| `--decision-model-floor` | `-dmt` | Ollama probability threshold (default: 0.5) |
| `--decision-model-sample-rate` | `-dmsr` | Frame sampling interval in seconds (default: 3) |
| `--ollama-host` | `-oh` | Ollama server address (default: `192.168.1.190`) |
| `--ollama-model` | `-dm` | Ollama model name (default: `typesafe/jev-1.14`) |
| `--ollama-frame-scale` | `-ofs` | Image scale factor (0.0–1.0, default: 0.1) |
| `--ollama-timeout` | `-ot` | Ollama request timeout in seconds (default: 60) |
| `--still-frame-threshold` | | Similarity above which a frame is considered still (default: 0.95) |
| `--same-scene-threshold` | | Similarity above which frames are same scene (default: 0.80) |
| `--merge-range` | | Merge adjacent same-scene frames into timestamp ranges |
| `--recursive` | `-R` | Recursively scan directories for video files |
| `--cache-dir` | | Directory for caching results (lookup if present, otherwise save) |
| `--ollama-classify-question` | `-cq` | Question name (default: `team`) |
| `--ollama-classify-instructions` | `-ci` | Instructions for the choice question |
| `--ollama-classify-criteria` | `-cc` | Criteria as JSON dict or multiline `key: description` |

## Yes/No (Search-Equivalent) Classification

To replicate the old `vsearch search` behavior that used the `noul` (yes/no)
question type, simply use a two-choice criteria:

```bash
vsearch classify myvideo.mp4 \
  --ollama-classify-question is_valid \
  --ollama-classify-instructions "Does this frame match the query?" \
  --ollama-classify-criteria "yes: the frame matches the query\nno: the frame does not match" \
  --top-choice yes --decision-model-floor 0.7
```

This is exactly equivalent to the old search mode with `ollama-search-question
is_valid` and `ollama-search-question-type noul` — the `choice` question type
with `yes`/`no` criteria handles binary judgment cleanly, and the unified cache
and filtering apply.

## Configuration (`config.toml`)

The config file contains a single `[classify]` section:

```toml
[classify]
top_k = -1
json_output = false
decision_model_floor = 0.5
decision_model_sample_rate = 3
ollama_host = "192.168.1.190"
ollama_frame_scale = 0.1
ollama_decision_model = "typesafe/jev-1.14"
ollama_timeout = 60
cache_dir = "/tmp"
still_frame_threshold = 0.95
same_scene_threshold = 0.80
merge_range = false
ollama_state_template = "sample frame from a video"
ollama_classify_question = "team"
ollama_classify_instructions = "Which team should handle this scene?"
ollama_classify_criteria = """
billing: Payments, payouts, invoices, refunds
technical: Bugs, outages, integrations, API errors
sales: Pricing, upgrades, new accounts
"""
```

CLI flags always override config values.

## Performance Tips

- **Scale down images** with `--ollama-frame-scale` (e.g. `0.15`) — lower = faster inference.
- **Same-scene batching** — consecutive frames with similarity ≥ `same_scene_threshold` are grouped and sent as a single multi-image request.
- **Coarser sampling** with `--decision-model-sample-rate 5` — one frame every 5 seconds.
- **Lower the threshold** with `--decision-model-floor 0.4` to keep more passing frames.
- **Use the cache** — re-running with different thresholds is instant on cache hit.

## Architecture

- **No indexing step** — videos are read directly with OpenCV.
- **No `.npz` cache** — frames are extracted and validated on the fly.
- **Ollama-only scoring** — `top_probability` from the choice question is the score.
- **Endpoint fallback** — if the configured Ollama host is unreachable, `localhost:11434` is used automatically.
- **Same-scene batching** — frames with high visual similarity within a short window are grouped and sent to Ollama as a single multi-image batch request (max 4 frames per batch).

## Classification Cache

The cache is a **SQLite database** stored on disk, enabling fast lookups,
concurrent reads (safe for AI agents), and custom SQL queries for advanced users.

### Unified `--cache-dir` Flag

The `--cache-dir` flag serves double duty:

- **If a matching cache entry exists** (same video content + same Ollama params) →
  results loaded instantly from disk (no Ollama calls).
- **If no match** → full classification runs, results saved to the SQLite
  database after completion.

```bash
# First run: full classification, saves to cache DB
vsearch classify video.mp4 --cache-dir /tmp/cache

# Second run: cache hit (instant, no Ollama)
vsearch classify video.mp4 --cache-dir /tmp/cache
```

### Cache Key (content + params)

The cache matches on **both** video content and Ollama parameters:

| Component | Description |
|-----------|-------------|
| `content_hash` | BLAKE2b of first/last 8 MB of video file |
| `params_hash` | BLAKE2b of `sample_rate \| frame_scale \| state \| instructions \| model \| criteria_json` |

If either changes, it's a cache miss. This ensures correctness when you change
sampling rate, frame scale, state, instructions, criteria, or model between runs.

### Cache Schema

The SQLite database contains three tables:

```sql
videos (
    id, video_path, content_hash, params_hash,
    fps, total_frames, stride, created_at, updated_at, cache_status
)

frames (
    id, video_id, frame_index, timestamp_seconds,
    top_choice, top_probability, probabilities_json,
    batch_id, raw_response, created_at
)

cache_stats (key, value, updated_at)
```

### Cache Management CLI

`vsearch cache` subcommand for managing the SQLite cache:

```bash
# List all cached videos
vsearch cache list --cache-dir /tmp/cache

# Show cache statistics (videos count, frames count)
vsearch cache stats --cache-dir /tmp/cache

# Prune entries older than N days (default 30)
vsearch cache prune 7 --cache-dir /tmp/cache

# Execute custom SQL query
vsearch cache sql "SELECT video_path, COUNT(*) AS n FROM frames GROUP BY video_path" --cache-dir /tmp/cache
```

### Cache Hit Behavior

- **Exact cache hit**: SQLite lookup matches video content + params → results loaded instantly, filtered by current `decision_model_floor`
- **Cache miss**: Video not in DB or params changed → full Ollama inference runs, results saved to DB
- **Params-aware**: Changing sample_rate, frame_scale, state, instructions, criteria, or model triggers a new run

### Multi-Video Classification

Cache works **per video file**:

- Each video in a multi-video run gets its own cache entry
- Cache-hit videos are skipped (no Ollama calls)
- Cache-miss videos are classified normally
- Results combined and sorted chronologically

```bash
# Example: video1 cached, video2 not
vsearch classify video1.mp4 video2.mp4 --cache-dir /tmp/cache
# -> video1: cache hit, video2: full classification

# Example: 5 videos all cached
vsearch classify /path/to/videos/ --cache-dir /tmp/cache
# -> all 5: cache hits
```

### AI Agent Usage

For AI agents using vsearch as a tool:

```bash
# Initialize/check cache DB
vsearch cache stats --cache-dir /path/to/cache

# Classify with caching
vsearch classify video.mp4 --cache-dir /path/to/cache

# Re-run with different floor (cache hit, no Ollama calls needed):
vsearch classify video.mp4 --cache-dir /path/to/cache --decision-model-floor 0.3

# Re-run filtering by a specific choice (cache hit):
vsearch classify video.mp4 --cache-dir /path/to/cache --top-choice technical

# Inspect what's cached
vsearch cache list --cache-dir /path/to/cache

# Custom SQL query on cache
vsearch cache sql "SELECT video_path, top_choice, COUNT(*) FROM frames GROUP BY video_path, top_choice" --cache-dir /path/to/cache

# Query frames by secondary choice probability (even if not the top choice)
vsearch cache sql "SELECT video_path, timestamp_seconds FROM frames WHERE json_extract(probabilities_json, '$.technical') > 0.5" --cache-dir /path/to/cache

# Prune old entries
vsearch cache prune 30 --cache-dir /path/to/cache  # keep last 30 days only
```