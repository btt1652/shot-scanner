# vsearch — Semantic Video Frame Analysis

A lightweight CLI that analyzes video frames using an Ollama decision model as the sole judge.

## CLI Subcommands

| Command | Description |
|---------|-------------|
| `vsearch analyze <video>` | Analyze video frames using Ollama choice-question model |
| `vsearch query "<sql>" [--cache CACHE] [--save-frame DIR]` | Query the cache or extract matching frames |
| `vsearch search <video> [--q "question=choice:prob"]* [--top N] [--model MODEL] [--run-id RUN_ID] [--save-frame DIR]` | Human-friendly search without SQL |
| `vsearch cache <cmd>` | Cache management: list, delete, prune, clear |

### Query Subcommand (`vsearch query`)

```bash
# List all cached videos (by id and path)
vsearch query "SELECT id, video_path FROM videos" --cache ./.cache.db

# Show cache statistics
vsearch query "SELECT COUNT(*) AS total_videos, SUM(answer_count) AS total_frames FROM (SELECT v.id, COUNT(fa.id) AS answer_count FROM videos v LEFT JOIN analysis_runs ar ON v.id = ar.video_id LEFT JOIN frame_answers fa ON ar.id = fa.run_id GROUP BY v.id)" --cache ./.cache.db

# Prune entries older than N days (default 30)
vsearch query "DELETE FROM analysis_runs WHERE updated_at < datetime('now', '-30 day')" --cache ./.cache.db

# Execute custom SQL query
vsearch query "SELECT * FROM frame_answers WHERE question_name='activity'" --cache ./.cache.db

# Extract and save frame images from videos based on cached answer filters
vsearch query "SELECT fa.run_id, fa.frame_index, fa.timestamp_seconds, fa.top_choice FROM frame_answers fa JOIN analysis_runs ar ON fa.run_id = ar.id WHERE fa.question_name='activity'" --save-frame ./activity_frames

# Filter by question + choice
vsearch query "SELECT fa.run_id, fa.frame_index, fa.timestamp_seconds, fa.top_choice FROM frame_answers fa JOIN analysis_runs ar ON fa.run_id = ar.id WHERE fa.question_name='activity' AND fa.top_choice='walking'" --save-frame ./walking_frames

# Only high-confidence frames
vsearch query "SELECT fa.run_id, fa.frame_index, fa.timestamp_seconds, fa.top_choice FROM frame_answers fa JOIN analysis_runs ar ON fa.run_id = ar.id WHERE fa.question_name='activity' AND fa.top_probability > 0.8" --save-frame ./high_conf_activity

# Filename format: video_name_f000001_12.3s_activity_walking.jpg
```

## Configuration (`config.toml`)

```toml
decision_model_sample_rate = 1
ollama_host = "localhost"
ollama_decision_model = "d1:3b"  # or "clef-flash:9b"
activity = { walking="Person standing", running="Person sitting", sitting="Person laying down" }
```

## Working Examples

- `python3 -m vsearch query "SELECT frame_index, timestamp_seconds, top_choice, top_probability FROM frame_answers WHERE top_probability > 0.9"`
- `python3 -m vsearch query "SELECT fa.frame_index, fa.timestamp_seconds, fa.top_choice FROM frame_answers fa WHERE fa.top_choice='walking' AND fa.top_probability > 0.8" --save-frame ./walking_frames`

## Frame Filenames

Filename format: `{video_name}_f{frame_index:06d}_{ts:.1f}s_{question_name}_{choice}.jpg`

## Human-Friendly Search

For interactive use, use the `search` command with simple key-value filters instead of raw SQL.

### Basic Search
```bash
# Find frames where activity is "yes" with probability >= 0.8
vsearch search video.mp4 --q "activity=yes:0.8"

# Find frames where time_of_day is "evening" with probability >= 0.9
vsearch search video.mp4 --q "time_of_day=evening:0.9"

# Multiple filters (AND logic - all must match)
vsearch search video.mp4 \
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

### Frame Extraction via Search
```bash
# Search and save frames
vsearch search video.mp4 \
  --q "activity=walking:0.8" \
  --q "time_of_day=morning:0.9" \
  --top 5 \
  --save-frame ./walking_frames
```

This will extract frames meeting all criteria, sorted by probability, capped at top N.

## Cache Management (`vsearch cache`)

For managing the SQLite cache directly.

```bash
# List all cached videos with answer counts
vsearch cache list

# Limit results
vsearch cache list --limit 10

# Delete a specific video and its answers
vsearch cache delete "/path/to/video.mp4"

# Prune videos older than N days (default 30)
vsearch cache prune --days 30

# Clear entire cache (requires --yes confirmation)
vsearch cache clear --yes
```

## Cache Schema

The cache is a SQLite database (`./.cache.db`) with the following schema:

- **videos**: Video metadata (path, content hash, fps, total frames)
- **analysis_runs**: Each analysis run for a video with different parameters (content_hash + params_hash unique constraint, model name)
- **frame_answers**: Per-frame answers for each run (run_id, frame_index, question_name, top_choice, top_probability)
- **cache_stats**: Key-value statistics

Each `(video, content_hash, params_hash)` combination gets its own `analysis_runs` entry, so different analysis configurations for the same video can coexist.