# scs — SCeneScan — Semantic Video Frame Analysis

A lightweight CLI that analyzes video frames using an Ollama decision model as the sole judge.

## CLI Subcommands

| Command | Description |
|---------|-------------|
| `scs analyze <video>` | Analyze video frames using Ollama choice-question model |
| `scs query "<sql>" [--cache CACHE] [--save-frame DIR]` | Query the cache or extract matching frames |
| `scs search <video> [--q "(group)"]* [--top N] [--top-per-video N] [--merge-result-threshold SEC] [--model MODEL] [--run-id RUN_ID] [--save-frame DIR] [--stitch]` | Human-friendly search with compact or grouped filter syntax (see below) |
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
ollama_frame_history = 5
merge_same_threshold = 0     # >0 enables -mst (same-score merging)
merge_threshold = 0          # >0 enables -mt (all-mode merging, mutually exclusive with -mst)
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
- `--ollama-frame-history N` → number of previous frames to use as context for current analysis (default: 5). Always uses **full JSON** format including all probabilities and confidence scores.
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

## Frame History Context

By default, `scs analyze` sends independent batches of frames to the Ollama model — each batch has no knowledge of what happened in previous scenes. The `ollama_frame_history` setting enables **temporal anchoring**: including recent previous frames' question answers as context for each new request.

### How it works

- Set `ollama_frame_history = N` (default: 5) in `config.toml`, or use `--ollama-frame-history N` on the CLI
- By default, the system always uses **full JSON format** to dump complete raw response body per question — including all probabilities, confidence scores, and metadata from the Ollama response for maximum fidelity
- Before sending each batch, the system builds a context string listing the N most recent processed frames with their timestamps and question answers
- Still frames detected between processed frames are tracked as `[still_frame]` (no model call was made), so the model knows about processing gaps
- This context is appended to the `state` field sent to Ollama, giving the model temporal awareness without adding extra images

### Example context output

The system always uses **full JSON format** to provide maximum visibility into previous frame results:

```
Recent context (3 frames):
Frame 120 (00:04:00):
  activity: {"choice": "yes", "probabilities": {"yes": 0.88, "no": 0.12}, "confidence": 0.91}
  act_category: {"choice": "oral_on_female", "probabilities": {"oral_on_female": 0.76, "vaginal_sex": 0.15, "non_sexual": 0.09}, "confidence": 0.82}
  time_of_day: {"choice": "evening", "probabilities": {"evening": 0.91, "morning": 0.06, "afternoon": 0.03}, "confidence": 0.95}
Frame 150 (00:05:00) [still_frame]: skipped
Frame 180 (00:06:01):
  activity: {"choice": "yes", "probabilities": {"yes": 0.85, "no": 0.15}, "confidence": 0.87}
  act_category: {"choice": "oral_on_female", "probabilities": {"oral_on_female": 0.72, "vaginal_sex": 0.19, "non_sexual": 0.09}, "confidence": 0.79}
  time_of_day: {"choice": "evening", "probabilities": {"evening": 0.89, "morning": 0.07, "afternoon": 0.04}, "confidence": 0.93}
```

This tells the model:
- At 00:04:00 the scene had high-confidence results (`yes: 0.88`, `evening: 0.91`)
- A still frame at 00:05:00 means the camera was static — no model call needed
- At 00:06:01 the pattern held consistently (`activity=yes: 0.85`), just slightly lower confidence after a gap

The full JSON response gives the model complete visibility into previous decisions: all probability distributions, confidence scores, and metadata. This allows detecting subtle shifts even when the top choice doesn't change (e.g., `vaginal_sex` went from 0.15 → 0.19), helping the model distinguish between stable scenes and transitions.

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
- `--q "question=choice:prob[:confidence]"` (`-q`): Filter condition (repeatable).

  **Compact syntax** (legacy): `question=choice:field>=val[&&field>=val...]`
  - Within a single `--q`, use `&&` for AND, `||` for OR: `activity=yes:prob>=0.8&&conf>=0.5` or `act_category=oral_on_female:prob>=0.6||conf>=0.7`
  - Multiple `--q` flags are AND'd together.

  **Grouped syntax**: Wrap complex conditions in parentheses to group independent questions:
  ```bash
  # OR between groups, AND within each group
  scs search video.mp4 --q '(act_category=oral_on_female:prob ge 0.8&&conf ge 0.5)||(act_category=vaginal_sex:prob ge 0.7)'
  
  # AND between multiple --q groups
  scs search video.mp4 --q '(act_category=oral_on_female:prob ge 0.6)' --q '(camera_angle=front:prob ge 0.7)'
  
  # Nested OR within a group
  scs search video.mp4 --q '(act_category=oral_on_female:prob ge 0.6||conf ge 0.7)'
  ```
  
  Group syntax lets each group specify its own `question=choice` prefix with independent field filters combined via `&&`/`||`. Groups at the same `--q` level are joined by `||` (OR); multiple `--q` flags are AND'd. Use `ge` (word operator) instead of `>=` to avoid shell escaping issues.
- `--top N` (`-t`): Limit to top N frames (or ranges with merge) by highest probability, then results are re‑ordered chronologically.
- `--top-per-video N` (`-tpv`): Limit number of top results per video by highest probability, then results are re‑ordered chronologically per video. When both `--top` and `--top-per-video` are supplied, `--top-per-video` takes precedence.
- `--merge-same-threshold SEC` (`-mst`): Merge consecutive frames **with the SAME probability score** within `SEC` seconds into a single time-range. Only merges rows sharing an identical `top_probability`. Default from `config.toml`: `merge_same_threshold = 0` (disabled). Set `> 0` to enable, or override via CLI.
- `--merge-threshold SEC` (`-mt`): Merge **ALL** adjacent query results into continuous time-ranges based on timestamp proximity alone (regardless of score differences). Produces time-range blocks spanning any probability scores as long as frames are within the threshold window. Default from `config.toml`: `merge_threshold = 0` (disabled). Set `> 0` to enable, or override via CLI. Mutually exclusive with `-mst` — both cannot be enabled simultaneously.

  **Example — two merge modes:**
  Given frames at 10.0s (prob=0.85), 10.5s (prob=0.85), 11.0s (prob=0.72), 11.5s (prob=0.72):
  - `-mst 1.0` → two ranges: `[10.0-10.5] prob=0.85` and `[11.0-11.5] prob=0.72` (score break at 11.0s)
  - `-mt 1.0` → one range: `[10.0-11.5]` best\_prob=0.85 (all merged by timestamp continuity)

  When used with `--top` or `--top-per-video`, the limit counts ranges (not individual frames).
- `--save-frame DIR` (`-sf`): Save extracted frames to directory (JPG only)
- `--model MODEL` (`-m`): Filter results by Ollama model name (e.g., clef-flash:9b, d1:3b)
- `--run-id RUN_ID` (`-ri`): Filter results to a specific analysis run ID
- `--cache CACHE` (`-c`): Path to SQLite cache file (default: ./.cache.db)
- `--stitch [PATH]` (`-st`): Stitch matched frames into a single video using ffmpeg. Default threshold is 1 second unless `--merge-same-threshold/-mst` or `--merge-threshold/-mt` is provided.

#### Quick Reference — Shorthand Flags
| Long Flag | Short | Default in config.toml | Description |
|-----------|-------|------------------------|-------------|
| `--top` | `-t` | — | Limit to top N results |
| `--top-per-video` | `-tpv` | — | Limit per video |
| `--merge-same-threshold` | `-mst` | `merge_same_threshold = 0` | Merge same-score frames (set >0 to enable) |
| `--merge-threshold` | `-mt` | `merge_threshold = 0` | Merge all frames by time (mutually exclusive with -mst) |
| `--save-frame` | `-sf` | — | Extract frame images |
| `--model` | `-m` | — | Filter by model |
| `--run-id` | `-ri` | — | Filter by run ID |
| `--cache` | `-c` | — | Cache path |
| `--stitch` | `-st` | — | Video stitching

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