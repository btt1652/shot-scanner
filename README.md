# scs — SCeneScan — Semantic Video Frame Analysis

A lightweight CLI that analyzes video frames using an Ollama decision model as the sole judge.

## CLI Subcommands

| Command | Description |
|---------|-------------|
| `scs analyze <video>` | Analyze video frames using Ollama choice-question model |
| `scs query "<sql>" [--cache CACHE] [--save-frame DIR]` | Query the cache or extract matching frames |
| `scs search <video> [--q "question=choice:prob"]* [--top N] [--top-per-video N] [--merge-result-threshold SEC] [--model MODEL] [--run-id RUN_ID] [--save-frame DIR] [--stitch]` | Human-friendly search without SQL |
| `scs cache <cmd>` | Cache management: list, delete, prune, clear, clean |

## isearch - Experimental Image-Based Search

Search cached frame answers by reference image (single image). Uses a single reference image to find semantically similar frames across cached analysis runs.

### Usage
```bash
scs isearch <image> [--question Q] [-qr RUN_ID|VIDEO] [--top N] [--top-per-video N] [--merge-result-threshold SEC] [--model MODEL] [--run-id RUN_ID] [--save-frame DIR] [--stitch] [--cache PATH]
```

### Parameters
- `image`: Path to reference image (jpg, png, webp, bmp …)
- `--question Q` or `-Q`: Question name to score the reference image against (optional if `--question-ref` is given)
- `-qr RUN_ID|VIDEO` or `--question-ref`: Use the question set from a run_id (integer) or latest run of a video path (string)
- `--top N`: Limit to top N frames by probability
- `--top-per-video N`: Limit number of top results per video by probability
- `--merge-result-threshold SEC`: Timestamp difference threshold (seconds) to merge adjacent frames with same score
- `--model MODEL`: Filter results by Ollama model name (e.g., clef-flash:9b, d1:3b)
- `--run-id RUN_ID`: Filter results to a specific analysis run ID
- `--save-frame DIR`: Directory to save extracted frames (JPG only)
- `--stitch [PATH]`: Stitch matched frames into a single video using ffmpeg. Uses --merge-result-threshold (default 1 sec) to group consecutive frames with same score into ranges before stitching
- `--cache PATH`: Path to SQLite cache file (default: ./.cache.db)

### Example
```bash
# Search using run_id (integer)
scs isearch frames/hddJUR-091_cut_f020677_689.9s_activity_standing.jpg --question-ref 1 --top 1

# Search using video path (string)
scs isearch frames/hddJUR-091_cut_f020677_689.9s_activity_standing.jpg -qr /mnt/hdd/hddJUR-091_cut.mp4 --top 1

# Search with explicit question name
scs isearch frames/hddJUR-091_cut_f020677_689.9s_activity_standing.jpg --question activity --top 1
```
### Query Subcommand (`scs query`)

```bash
# List all cached videos (by id and path)
scs query "SELECT id, video_path FROM videos" --cache ./.cache.db

# Show cache statistics
scs query "SELECT COUNT(*) AS total_videos, SUM(answer_count) AS total_frames FROM (SELECT v.id, COUNT(fa.id) AS answer_count FROM videos v LEFT JOIN analysis_runs ar ON v.id = ar.video_id LEFT JOIN frame_answers fa ON ar.id = fa.run_id GROUP BY v.id)" --cache ./.cache.db

# Prune entries older than N days (default 30)
scs query "DELETE FROM analysis_runs WHERE updated_at < datetime('now', '-30 day')" --cache ./.cache.db

# Execute custom SQL query
scs query "SELECT * FROM frame_answers WHERE question_name='activity'" --cache ./.cache.db

# Extract and save frame images from videos based on cached answer filters
scs query "SELECT fa.run_id, fa.frame_index, fa.timestamp_seconds, fa.top_choice FROM frame_answers fa JOIN analysis_runs ar ON fa.run_id = ar.id WHERE fa.question_name='activity'" --save-frame ./activity_frames

# Filter by question + choice
scs query "SELECT fa.run_id, fa.frame_index, fa.timestamp_seconds, fa.top_choice FROM frame_answers fa JOIN analysis_runs ar ON fa.run_id = ar.id WHERE fa.question_name='activity' AND fa.top_choice='walking'" --save-frame ./walking_frames

# Only high-confidence frames
scs query "SELECT fa.run_id, fa.frame_index, fa.timestamp_seconds, fa.top_choice FROM frame_answers fa JOIN analysis_runs ar ON fa.run_id = ar.id WHERE fa.question_name='activity' AND fa.top_probability > 0.8" --save-frame ./high_conf_activity

# Filename format: video_name_f000001_12.3s_activity_walking.jpg
```

## Configuration (`config.toml`)

The `config.toml` file uses a **flat** structure — no section headers. It provides default values; command-line arguments always override them at runtime.

```toml
# Global defaults
decision_model_sample_rate = 1
ollama_host = "192.168.1.190"
ollama_frame_scale = 0.1
ollama_decision_model = "d1:3b"
ollama_timeout = 60
ollama_state_template = "sample frame from a video"
cache = "./.cache.db"
still_frame_threshold = 0.95
same_scene_threshold = 0.80
recursive = false

# Question definitions (one [[questions]] entry per question)
[[questions]]
name = "activity"
instructions = "What activity is happening in this frame?"
criteria = { walking = "Person walking", running = "Person running", sitting = "Person sitting" }

[[questions]]
name = "time_of_day"
instructions = "What time of day is it?"
criteria = { morning = "Morning light", afternoon = "Afternoon light", evening = "Evening light" }
```

### Override Precedence
Command-line arguments take precedence over `config.toml`:
- `--ollama-model MODEL` → overrides `ollama_decision_model` used for new analysis runs (analyze only)
- `--model MODEL` → filters results by model name when querying cache (search only)
- `--q "question=choice:prob"` → adds filter criteria (search only)
- `--run-id RUN_ID` → targets a specific analysis run ID (search only)
- `--cache PATH` → overrides the default `.cache.db` location

**Example**: Running `scs analyze video.mp4 --ollama-model clef-flash:9b` uses `clef-flash:9b` for new frames; running `scs search video.mp4 --model clef-flash:9b` filters existing cached answers to that model.

## Working Examples

- `python3 -m scs query "SELECT frame_index, timestamp_seconds, top_choice, top_probability FROM frame_answers WHERE top_probability > 0.9"`
- `python3 -m scs query "SELECT fa.frame_index, fa.timestamp_seconds, fa.top_choice FROM frame_answers fa WHERE fa.top_choice='walking' AND fa.top_probability > 0.8" --save-frame ./walking_frames`

## Frame Filenames

Filename format: `{video_name}_f{frame_index:06d}_{ts:.1f}s_{question_name}_{choice}.jpg`

## Human-Friendly Search

For interactive use, use the `search` command with simple key-value filters instead of raw SQL.

### Basic Search
```bash
# Find frames where activity is "yes" with probability >= 0.8
scs search video.mp4 --q "activity=yes:0.8"

# Find frames where time_of_day is "evening" with probability >= 0.9
scs search video.mp4 --q "time_of_day=evening:0.9"

# Multiple filters (AND logic - all must match)
scs search video.mp4 \
  --q "activity=yes:0.8" \
  --q "time_of_day=evening:0.9" \
  --top 10 \
  --save-frame ./frames
```

### Parameters
- `<video>`: Video path (or glob/directory pattern)
- `--q "question=choice:prob[:confidence]"`: Filter condition (repeatable). Add optional confidence threshold (e.g., `activity=yes:0.8:0.7`). **Within a single `--q`, use ` | ` for OR logic** (e.g., `activity=yes:0.8 | activity=no:0.7`). **Multiple `--q` flags are AND'd together**.
  - Example: `--q "activity=yes:0.8 | activity=no:0.7" --q "scene_type=couple:0.9"` means `(activity=yes OR activity=no) AND scene_type=couple`.
- `--top N`: Limit to top N frames (or ranges with `--merge-result-threshold`) by highest probability, then results are re‑ordered chronologically.
- `--top-per-video N`: Limit number of top results per video by highest probability, then results are re‑ordered chronologically per video. When both `--top` and `--top-per-video` are supplied, `--top-per-video` takes precedence.
- `--merge-result-threshold SEC`: Merge consecutive frames with the same score within `SEC` seconds into a single range. When used with `--top` or `--top-per-video`, the limit counts ranges (not individual frames).
- `--save-frame DIR`: Save extracted frames to directory (JPG only)
- `--model MODEL`: Filter results by Ollama model name (e.g., clef-flash:9b, d1:3b)
- `--run-id RUN_ID`: Filter results to a specific analysis run ID
- `--cache CACHE`: Path to SQLite cache file (default: ./.cache.db)
- `--stitch [PATH]`: Stitch matched frames into a single video using ffmpeg. Uses `--merge-result-threshold` (default 1 sec) to group consecutive frames with the same score into ranges before stitching.

### Frame Extraction via Search
```bash
# Search and save frames
scs search video.mp4 \
  --q "activity=walking:0.8" \
  --q "time_of_day=morning:0.9" \
  --top 5 \
  --save-frame ./walking_frames
```

This will extract frames meeting all criteria and return them **sorted by timestamp (ascending) by default**. If you also use `--top` or `--top-per-video`, the rows are first selected by highest probability and then re‑ordered chronologically.

## Cache Table Schema (Visual)

Understanding the table structure helps you write better custom SQL queries. Below is a visual representation of each table's columns and their purposes.

### 📋 `videos` Table
| Column | Type | Description |
|--------|------|-------------|
| `id` | INTEGER | Primary key |
| `video_path` | TEXT | Full path to the video file |
| `content_hash` | TEXT | SHA256 hash of file content (for dedup/symlink handling) |
| `created_at` | TEXT | Timestamp when video was first cached |

### 📊 `analysis_runs` Table
| Column | Type | Description |
|--------|------|-------------|
| `id` | INTEGER | Primary key (run ID) |
| `video_id` | INTEGER | FK → videos.id |
| `params_hash` | TEXT | Hash of model+config params (question_name+params) |
| `model` | TEXT | Ollama model name (e.g., `clef-flash:9b`) |
| `questions_json` | TEXT | Full questions array as JSON (for reproducibility) |
| `cache_status` | TEXT | `empty`/`partial`/`complete` |
| `created_at` | TEXT | Run creation timestamp |
| `updated_at` | TEXT | Last update timestamp |

### 🖼️ `frame_answers` Table
| Column | Type | Description |
|--------|------|-------------|
| `run_id` | INTEGER | FK → analysis_runs.id |
| `frame_index` | INTEGER | Frame index within video |
| `timestamp_seconds` | REAL | Timestamp in seconds |
| `question_name` | TEXT | Question key (e.g., `activity`, `time_of_day`) |
| `top_choice` | TEXT | Best answer (e.g., `walking`, `sitting`) |
| `top_probability` | REAL | Top choice probability (0.0–1.0) |
| `confidence` | REAL | Model confidence/certainty (0.0–1.0, from Ollama) |
| `raw_response` | TEXT | Raw Ollama response JSON |

### 📈 `cache_stats` Table
| Column | Type | Description |
|--------|------|-------------|
| `key` | TEXT | Statistic key |
| `value` | TEXT | Statistic value |

---

## Cache Management (`scs cache`)

For managing the SQLite cache directly.

```bash
# List all cached videos with answer counts
scs cache list

# Limit results
scs cache list --limit 10

# Delete a specific video or run_id
scs cache delete "/path/to/video.mp4"
scs cache delete 7  # Delete specific run_id (integer)

# Prune videos older than N days (default 30)
scs cache prune --days 30

# Clear entire cache (requires --yes confirmation)
scs cache clear --yes

# Remove cache entries for missing or changed videos
scs cache clean
```

## Stitching Extracted Frames

The `--stitch` flag uses ffmpeg to merge adjacent frames into a single video file.

**Usage:**

```bash
# Stitch matched frames (auto-generates output: {video_name}_stitched.mp4)
scs search video.mp4 \
  --q "activity=sitting:0.95" \
  --top 3 \
  --stitch

# Stitch to custom output path
scs search video.mp4 \
  --q "activity=sitting:0.95" \
  --stitch custom_output.mp4
```