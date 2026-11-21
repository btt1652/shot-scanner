# scs — SCeneScan — Semantic Video Frame Analysis

A lightweight CLI that analyzes video frames using an Ollama decision model as the sole judge.

## CLI Subcommands

| Command | Description |
|---------|-------------|
| `scs analyze <video>` | Analyze video frames using Ollama choice-question model |
| `scs query "<sql>" [--cache CACHE] [--save-frame DIR]` | Query the cache or extract matching frames |
| `scs search <video> [--q "question=choice:prob"]* [--top N] [--model MODEL] [--run-id RUN_ID] [--save-frame DIR] [--stitch] [--stitch-merge-threshold SEC]` | Human-friendly search without SQL |
| `scs cache <cmd>` | Cache management: list, delete, prune, clear |

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

The `config.toml` file provides **default** configuration values. Command-line arguments **override** these defaults at runtime.

```toml
decision_model_sample_rate = 1
ollama_host = "localhost"
ollama_decision_model = "d1:3b"  # or "clef-flash:9b"
activity = { walking="Person standing", running="Person sitting", sitting="Person laying down" }
```

### Override Precedence
Command-line arguments take precedence over `config.toml`:
- `--model MODEL` → overrides `ollama_decision_model` in config
- `--q "question=choice:prob"` → adds filter criteria (not in config)
- `--run-id RUN_ID` → targets specific analysis run (not in config)
- `--cache PATH` → overrides default `.cache.db` location

**Example**: If `config.toml` has `ollama_decision_model = "d1:3b"` but you run `scs search video.mp4 --model clef-flash:9b`, the search uses `clef-flash:9b`.

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
- `--q "question=choice:prob"`: Filter condition (repeatable, AND logic)
- `--top N`: Limit to top N frames by probability (default: all)
- `--save-frame DIR`: Save extracted frames to directory (JPG only)
- `--model MODEL`: Filter results by Ollama model name (e.g., clef-flash:9b, d1:3b)
- `--run-id RUN_ID`: Filter results to a specific analysis run ID
- `--cache CACHE`: Path to SQLite cache file (default: ./.cache.db)
- `--stitch [PATH]`: Stitch matched frames into a single video using ffmpeg. Optional: provide output path (auto-generates `{video_name}_stitched.mp4` if omitted).
- `--stitch-merge-threshold SEC`: Timestamp gap threshold (seconds) to merge adjacent frames (default: 1)

### Frame Extraction via Search
```bash
# Search and save frames
scs search video.mp4 \
  --q "activity=walking:0.8" \
  --q "time_of_day=morning:0.9" \
  --top 5 \
  --save-frame ./walking_frames
```

This will extract frames meeting all criteria, sorted by probability, capped at top N.

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
| `content_hash` | TEXT | Video content hash |
| `params_hash` | TEXT | Hash of model+config params (question_name+params) |
| `model` | TEXT | Ollama model name (e.g., `clef-flash:9b`) |
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
| `top_probability` | REAL | Confidence score (0.0–1.0) |
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

## Stitching Extracted Frames

The `--stitch` flag uses ffmpeg to merge adjacent frames into a single video file.

**Usage:**

```bash
# Stitch matched frames (auto-generates output: {video_name}_stitched.mp4)
scs search video.mp4 \
  --q "activity=sitting:0.95" \
  --top 3 \
  --stitch \
  --stitch-merge-threshold 2.0  # merge frames within 2 seconds

# Stitch to custom output path
scs search video.mp4 \
  --q "activity=sitting:0.95" \
  --stitch custom_output.mp4
```
scs cache clear --yes
```
