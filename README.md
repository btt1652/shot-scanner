# vsearch - Decision Model Video Search

A lightweight command-line interface that searches local video files using an Ollama decision model as the sole judge. No CLIP embeddings or pre-built `.npz` caches are required — frames are extracted and validated on the fly.

## Quick Start

1. Install dependencies:

   ```
   pip install -r requirements.txt
   ```

2. Search a video with a text prompt:

   ```
   vsearch search myvideo.mp4 --ollama-state "A user is searching for: a person running" --ollama-instructions "Does the image match the query: a person running?" -k 5
   ```

3. Limit the number of results:

   ```
   vsearch search myvideo.mp4 --ollama-state "A user is searching for: a person running" --ollama-instructions "Does the image match the query: a person running?" -k 5
   ```

4. Save matched frames as JPEG images:

   ```
   vsearch search myvideo.mp4 --ollama-state "A user is searching for: a person running" --ollama-instructions "Does the image match the query: a person running?" -sf /tmp/frames
   ```

5. Output JSON:

   ```
   vsearch search myvideo.mp4 --ollama-state "A user is searching for: a person running" --ollama-instructions "Does the image match the query: a person running?" --json
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
| `--top-k` | `-k` | Maximum number of top results to return (default: all passing) |
| `--global-k` | `-gk` | Global top-K across all videos, overriding `--top-k` |
| `--top-choice` | `-tc` | Only keep results where `top_choice` matches this value. Comma-separated for multiple choices (e.g., `technical` or `billing,technical`). Defaults to all choices. |
| `--decision-model-floor` | `-dmt` | Ollama score threshold (default: 0.5) |
| `--json` | | Output results as strict JSON |
| `--save-frames` | `-sf` | Directory to save matched frame images as {video_name}_{HHMMSS}_{score}_{norm_score}.jpg |
| `--decision-model-floor` | `-dmt` | Ollama score threshold (default: 0.5) |
| `--decision-model-sample-rate` | `-dmsr` | Frame sampling interval in seconds (default: 1) |
## Quick Start
| `--ollama-host` | `-oh` | Ollama server address (default: `192.168.1.190`). Falls back to `localhost` if unreachable |
| `--ollama-frame-scale` | `-ofs` | Scale factor for frames sent to Ollama (0.0–1.0, default: 0.2). Lower = faster |
| `--ollama-model` | `-dm` | Ollama model name (single model, no fallback) |
| `--recursive` | `-R` | Recursively scan directories for video files |
| `--ollama-search-question` | `-csq` | Search question name (default: `is_valid`) |
| `--ollama-search-question-type` | `-csqt` | Search question type (default: `noul`) |
| `--cache-dir` | | Directory for caching search results (lookup if present, otherwise save). Overrides `--cache` and `--save-cache`.

## Configuration (`config.toml`)

The config file contains a single `[search]` section:

```toml
[search]
top_k = 5
json_output = false
save_frames = ""
decision_model_floor = 0.8
decision_model_sample_rate = 3
ollama_host = "192.168.1.190"
ollama_frame_scale = 0.2
ollama_decision_model = "clef-flash:9b"
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
- `ollama_decision_model` — model name (default: `"clef-flash:9b"`)
- `ollama_state_template` — state context string sent to Ollama (default: empty)
- `ollama_instructions_template` — instructions string sent to Ollama (default: empty)
- `ollama_search_question` — search question name (default: `is_valid`)
- `ollama_search_question_type` — search question type (default: `noul`)
- `ollama_top_choice` — comma-separated choice(s) to keep in classify results (default: empty = keep all)

## Performance Tips

- **Scale down images** with `--ollama-frame-scale` (e.g. `0.15`) — lower = faster inference.
- **Same-scene batching** — consecutive frames with similarity ≥ `same_scene_threshold` (default `0.8`) are grouped into batches (max 4 frames per batch, constrained by Ollama's multi-image limit) and sent to Ollama as a single multi-image request.
- **Coarser sampling** with `--decision-model-sample-rate 5` — one frame every 5 seconds.
- **Lower the threshold** with `--decision-model-floor 0.4` to keep more passing frames.

## Architecture

- **No indexing step** — videos are read directly with OpenCV.
- **No `.npz` cache** — frames are extracted and validated on the fly.
- **Ollama-only scoring** — `final_score = Ollama_score`; CLIP is never invoked.
- **Endpoint fallback** — if the configured Ollama host is unreachable, `localhost:11434` is used automatically.
- **Same-scene batching** — frames with high visual similarity within a short window are grouped and sent to Ollama as a single multi-image batch request (max 4 frames per batch, constrained by Ollama's multi-image limit).


## Search Cache

The cache is a **SQLite database** stored on disk, enabling fast lookups, concurrent reads (safe for AI agents), and custom SQL queries for advanced users.

### Unified `--cache-dir` Flag

The `--cache-dir` flag serves double duty:

- **If a matching cache entry exists** (same video content + same Ollama params) -> results loaded instantly from disk (no Ollama calls).
- **If no match** -> full search runs, results saved to the SQLite database after completion.

```bash
# First run: full search, saves to cache DB
vsearch search video.mp4 --ollama-state "A user is searching for: a person" --ollama-instructions "Does the image match the query: a person?" --cache-dir /tmp/cache

# Second run: cache hit (instant, no Ollama)
vsearch search video.mp4 --ollama-state "A user is searching for: a person" --ollama-instructions "Does the image match the query: a person?" --cache-dir /tmp/cache
```

### Cache Key (content + params)

The cache matches on **both** video content and Ollama parameters:

| Component | Description |
|-----------|-------------|
| `content_hash` | BLAKE2b of first/last 8 MB of video file |
| `params_hash` | BLAKE2b of `sample_rate | frame_scale | state | instructions | model` |

If either changes, it's a cache miss. This ensures correctness when you change sampling rate, frame scale, state, instructions, or model between runs.

### Cache Management CLI

New `vsearch cache` subcommand for managing the SQLite cache:

```bash
# List all cached videos
vsearch cache list --cache-dir /tmp/cache

# Show cache statistics (videos count, frames count)
vsearch cache stats --cache-dir /tmp/cache

# Prune entries older than N days (default 30)
vsearch cache prune 7 --cache-dir /tmp/cache  # keep only last 7 days

# Execute custom SQL query
vsearch cache sql "SELECT video_path, COUNT(*) AS frame_count FROM frames GROUP BY video_path" --cache-dir /tmp/cache
```

### Cache Hit Behavior

- **Exact cache hit**: SQLite lookup matches video content + params -> results loaded instantly, filtered by current `decision_model_floor` threshold
- **Cache miss**: Video not in DB or params changed -> full Ollama inference runs, results saved to DB
- **Params-aware**: The database stores `content_hash` and `params_hash` -- changing sample_rate, frame_scale, state, instructions, or model triggers a new run

### Multi-Video Search

Cache works **per video file**:

- Each video in a multi-video search gets its own cache entry
- Cache-hit videos are skipped (no Ollama calls)
- Cache-miss videos are searched normally
- Results combined and sorted chronologically

```bash
# Example: video1 cached, video2 not
vsearch search video1.mp4 video2.mp4 --ollama-state "..." --ollama-instructions "..." --cache-dir /tmp/cache
# -> video1: cache hit, video2: full search

# Example: 5 videos all cached
vsearch search /path/to/videos/ --ollama-state "..." --ollama-instructions "..." --cache-dir /tmp/cache
# -> all 5: cache hits
```

### AI Agent Usage

For AI agents using vsearch as a tool:

```bash
# Initialize/check cache DB
vsearch cache stats --cache-dir /path/to/cache

# Search with caching
vsearch search video.mp4 --ollama-state "..." --ollama-instructions "..." --cache-dir /path/to/cache

# Re-run with different floor (cache hit, no Ollama calls needed):
vsearch search video.mp4 --ollama-state "..." --ollama-instructions "..." --cache-dir /path/to/cache --decision-model-floor 0.3

# Inspect what's cached
vsearch cache list --cache-dir /path/to/cache

# Custom SQL query on cache
vsearch cache sql "SELECT video_path, COUNT(*) AS frame_count FROM frames GROUP BY video_path" --cache-dir /path/to/cache

# Prune old entries
vsearch cache prune 30 --cache-dir /path/to/cache  # keep last 30 days only
```
