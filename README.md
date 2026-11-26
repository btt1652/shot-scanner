# scs — SCeneScan — Semantic Video Frame Analysis

A lightweight CLI that analyzes video frames using a decision model as the sole judge. Supports **Ollama** (local) or **OpenRouter** (cloud API).

## CLI Subcommands

| Command | Description |
|---------|-------------|
| `scs analyze <video>` | Analyze video frames using a decision model (Ollama or OpenRouter) |
| `scs query "<sql>" [--cache CACHE] [--save-frame DIR]` | Query the cache or extract matching frames |
| `scs search <video> [--q "(group)"]* [--top N] [--top-per-video N] [--merge-result-threshold SEC] [--model MODEL] [--run-id RUN_ID] [--save-frame DIR] [--stitch]` | Human-friendly search with compact or grouped filter syntax |
| `scs cache <cmd>` | Cache management: list, delete, prune, clear, clean |

## Provider: Ollama vs OpenRouter

Set the `provider` key in `config.toml` or use `--provider ollama` / `--provider openrouter` on the CLI:

| Setting | Ollama (default) | OpenRouter |
|---|---|---|
| **Where model runs** | Local machine | Cloud (openrouter.ai) |
| **Config key** | `ollama_host`, `ollama_timeout` | `openrouter_api_key`, `openrouter_model` |
| **Endpoint** | `http://host:11434/v1/systemone` | `https://openrouter.ai/api/alpha/decisions` |
| **Auth** | None | `Authorization: Bearer <key>` |
| **API key env var** | N/A | `OPENROUTER_API_KEY` |
| **Image count limit** | 4 per request | Varies by model (see below) |

All other settings (`frame_scale`, `state_template`, `frame_history`, questions, etc.) are shared.

### Ollama Quick Start

```toml
provider = "ollama"

[ollama]
host = "192.168.1.190"
```

```bash
scs analyze video.mp4          # uses local Ollama at 192.168.1.190
scs analyze video.mp4 -oh localhost   # override host for this run
```

### OpenRouter Quick Start

```toml
provider = "openrouter"
openrouter_api_key = "sk-..."      # or set OPENROUTER_API_KEY env var
openrouter_model = "perplexity/pplx-decider-v1.3"
```

```bash
# Via config.toml
scs analyze video.mp4

# Or via CLI flags (overrides config)
scs analyze video.mp4 \
  --provider openrouter \
  --openrouter-api-key sk-or-v1-xxx \
  --openrouter-model perplexity/pplx-decider-v1.3
```

The `OPENROUTER_API_KEY` environment variable is supported as a fallback when `openrouter_api_key` is not set in `config.toml`.

### How It Works

Both providers use the [Decisions API](https://openrouter.ai/docs/guides/features/decisions): they accept images + text `state` and return typed question answers (`choice`, `noul`, `score`) with probabilities and confidence scores. The response format is identical, so all downstream processing (caching, merging, search) works the same regardless of provider.

## Configuration (`config.toml`)

The `config.toml` file uses a structured format with `[ollama]` and `[openrouter]` sections for provider-specific settings. Provider-agnostic settings live at the top level. It provides default values; command-line arguments always override them at runtime.

```toml
# Decision engine: "ollama" (local) or "openrouter" (cloud API)
provider = "ollama"

# --- Provider-agnostic settings (used by both Ollama and OpenRouter) ---
frame_scale = 0.1                # Scale factor for frames sent to model (0.0–1.0)
decision_model = "d1:3b"         # Model name
state_template = ""               # Context string prepended to every request
frame_history = 5                 # Number of prior frames to include as context

# --- Ollama-specific settings ---
[ollama]
host = "192.168.1.190"           # Ollama server address
timeout = 60                      # Request timeout in seconds

# --- OpenRouter-specific settings ---
[openrouter]
api_key = ""                      # Set here or via OPENROUTER_API_KEY env var
model = "perplexity/pplx-decider-v1.3"
endpoint = "https://openrouter.ai/api/alpha/decisions"
timeout = 60

# --- Batch sizing ---
max_images_per_request = -1       # -1 = auto from model; >0 overrides (capped by model limit)

# --- Scene detection & caching ---
frame_sample_rate = 1
cache = "./.cache.db"
still_frame_threshold = 0.95
same_scene_threshold = 0.80
recursive = false

# --- Merge thresholds (search output grouping) ---
merge_same_threshold = 0     # >0 enables -mst (merge same-score frames)
merge_threshold = 0          # >0 enables -mt (merge adjacent frames)

isolated_frame_time = 2        # Padding for stitching isolated frame groups

# Per-model image limits for OpenRouter Decisions API.
# Edit to override defaults or add new models. Unlisted models fall back
# to built-in defaults (_BUILTIN_MODEL_LIMITS in scs/cli.py).
[model_image_limits]
  "perplexity/pplx-decider-v1.3" = 8
  "perplexity/pplx-decider-flash-v1.3" = 8
  "typesafe/jev-1.14" = 4
  "typesafe/jev-1.13" = 4
  "cloudflare/clef-flash" = 4
  "cloudflare/clef-omni" = 4
  "microsoft/microsoft-decision-1.1" = 8
  "google/gemini-decision-1" = 128
  "heygen/heygen-decision-1" = 100
  "liquid/d1.2" = 16
  "inception/mercury-decide" = 16
  "baseten/groundhog-1" = 16

# --- Question definitions (one [[questions]] entry per question) ---
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

| Flag | Overrides | Example |
|---|---|---|
| `--provider` | Provider selection | `--provider openrouter` |
| `--decision-model MODEL` | Decision model name | `--decision-model clef-flash:9b` |
| `--ollama-host HOST` | Ollama server address | `--ollama-host localhost` |
| `--ollama-timeout N` | Request timeout | `--ollama-timeout 120` |
| `--state-template TEXT` | State context string | `--state-template "frame from X"` |
| `--frame-history N` | Frame context window | `--frame-history 10` |
| `--frame-scale SCALE` | Frame scale factor | `--frame-scale 0.1` |
| `--openrouter-api-key SK` | OpenRouter API key | `--openrouter-api-key sk-...` |
| `--openrouter-model MODEL` | OpenRouter model | `--openrouter-model typesafe/jev-1.14` |
| `--openrouter-endpoint URL` | OpenRouter endpoint | `--openrouter-endpoint https://...` |
| `--openrouter-timeout N` | OpenRouter timeout | `--openrouter-timeout 120` |
| `--max-images-per-request N, -mipr N` | Batch size limit | `--mipr 6` |
| `--frame-sample-rate N` | Frame sampling rate | `--fsr 2` |
| `--model MODEL` | Cache query filter | `--model clef-flash:9b` |
| `--q "question=choice:prob"` | Search filter | `--q "activity=yes:0.8"` |
| `--run-id RUN_ID` | Target specific run | `--run-id 42` |
| `--cache PATH` | Cache database path | `--cache ./custom.cache.db` |

**Example**: Running `scs analyze video.mp4 --provider openrouter --openrouter-model typesafe/jev-1.14` uses OpenRouter with Jev for this run; running `scs search video.mp4 --model typesafe/jev-1.14` filters cached results to that model.

## Max Images Per Request

Batches are grouped by scene similarity and capped at a maximum number of images per request.

### Auto-detection (default)

When `max_images_per_request = -1` (the default), the system auto-detects the limit based on the active model:

| Model | Max Images |
|---|---|
| `perplexity/pplx-decider-v1.3`, `-flash-v1.3` | 8 |
| `typesafe/jev-*`, `cloudflare/clef-flash`, `clef-omni` | 4 |
| `microsoft/microsoft-decision-1.1` | 8 |
| `liquid/d1.*`, `inception/mercury-decide` | 16 |
| `heygen/heygen-decision-*` | 100 |
| `google/gemini-decision-1` | 128 |
| Ollama (any model) | 4 |
| Unknown / custom | 4 (safe fallback) |

### Configurable in `[model_image_limits]`

Edit `config.toml` under `[model_image_limits]` to override any model's limit or add new ones:

```toml
[model_image_limits]
  "perplexity/pplx-decider-v1.3" = 8
  "my-custom-model:latest" = 10    # new model
```

Unlisted models still fall back to the built-in defaults. You can also set a single global cap for all models:

```toml
# Cap ALL models at 5 frames per batch
max_images_per_request = 5
```

### Hard override via CLI

Force a specific limit for one run (still capped by the model's native provider limit):

```bash
scs analyze video.mp4 --mipr 6
```

**Priority chain** (highest wins):
1. `--max-images-per-request N` flag
2. `max_images_per_request = N` in `config.toml`
3. Auto-detection from `[model_image_limits]` + built-in defaults
4. Fallback to 4

The value is always **capped** by the model's native provider limit — you cannot exceed what the API allows.

## Using OpenRouter as Decision Provider

Instead of a local Ollama instance, you can use OpenRouter's cloud API. OpenRouter provides access to dozens of decision models including `perplexity/pplx-decider-v1.3`, `typesafe/jev-1.14`, `google/gemini-decision-1`, and many more.

### Setup

```toml
provider = "openrouter"
openrouter_api_key = "sk-or-v1-..."   # Create one at https://openrouter.ai/settings/keys
openrouter_model = "perplexity/pplx-decider-v1.3"
```

Or set the env var instead:
```bash
export OPENROUTER_API_KEY=sk-or-v1-...
```

### Endpoint

Requests go to `POST https://openrouter.ai/api/alpha/decisions` with:
- `Authorization: Bearer <api_key>` header
- Images embedded in the `state` array as `image_url` objects
- Same `answers` response format with `probabilities` and `confidence` fields

## Working Examples

```bash
# Analyze with Ollama (local, default)
python3 -m scs analyze video.mp4

# Analyze with OpenRouter
python3 -m scs analyze video.mp4 --provider openrouter --openrouter-api-key sk-...

# Analyze with specific OpenRouter model
python3 -m scs analyze video.mp4 --provider openrouter --openrouter-model typesafe/jev-1.14

# Search with filters
scs search video.mp4 --q "activity=yes:0.8" --q "time_of_day=evening:0.9"

# Search with JSON output
scs search video.mp4 --q "activity=yes:0.8" --top 3 --json

# Extract frames meeting criteria
scs search video.mp4 --q "activity=walking:0.8" --top 5 --save-frame ./walking_frames

# Stitch matched frames into a video
scs search video.mp4 --q "activity=sitting:0.95" --stitch

# Custom SQL query
python3 -m scs query "SELECT frame_index, timestamp_seconds, top_choice FROM frame_answers WHERE top_probability > 0.9" --save-frame ./high_confidence

# Image-based search
scs isearch reference.jpg --question activity --top 10
```

## Frame Filenames

Filename format: `{video_name}_f{frame_index:06d}_{ts:.1f}s_{question_name}_{choice}.jpg`

## Frame History Context

By default, `scs analyze` sends independent batches of frames to the decision model — each batch has no knowledge of what happened in previous scenes. The `frame_history` setting enables **temporal anchoring**: including recent previous frames' question answers as context for each new request.

### How it works

- Set `frame_history = N` (default: 5) in `config.toml`, or use `--frame-history N` on the CLI
- Before sending each batch, the system builds a context string listing the N most recent processed frames with their timestamps and question answers
- Still frames detected between processed frames are tracked as `[still_frame]` (no model call was made), so the model knows about processing gaps
- This context is appended to the `state` field sent to the model, giving it temporal awareness without adding extra images

### Example context output

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

# Multiple filters (AND logic — all must match)
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
  - Within a single `--q`, use `&&` for AND, `\|\|` for OR: `activity=yes:prob>=0.8&&conf>=0.5` or `act_category=oral_on_female:prob>=0.6\|\|conf>=0.7`
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
  
  Group syntax lets each group specify its own `question=choice` prefix with independent field filters combined via `&&`/`\|\|`. Groups at the same `--q` level are joined by `\|\|` (OR); multiple `--q` flags are AND'd. Use `ge` (word operator) instead of `>=` to avoid shell escaping issues.
- `--top N` (`-t`): Limit to top N frames (or ranges with merge) by highest probability, then results are re‑ordered chronologically.
- `--top-per-video N` (`-tpv`): Limit number of top results per video by highest probability, then results are re‑ordered chronologically per video. When both `--top` and `--top-per-video` are supplied, `--top-per-video` takes precedence.
- `--merge-same-threshold SEC` (`-mst`): Merge consecutive frames **with the SAME probability score** within `SEC` seconds into a single time-range. Only merges rows sharing an identical `top_probability`. Default from `config.toml`: `merge_same_threshold = 0` (disabled). Set `> 0` to enable, or override via CLI.
- `--merge-threshold SEC` (`-mt`): Merge **ALL** adjacent query results into continuous time-ranges based on timestamp proximity alone (regardless of score differences). Produces time-range blocks spanning any probability scores as long as frames are within the threshold window. Default from `config.toml`: `merge_threshold = 0` (disabled). Set `> 0` to enable, or override via CLI. Mutually exclusive with `-mst` — both cannot be enabled simultaneously.

  **Example — two merge modes:**
  Given frames at 10.0s (prob=0.85), 10.5s (prob=0.85), 11.0s (prob=0.72), 11.5s (prob=0.72):
  - `-mst 1.0` → two ranges: `[10.0-10.5] prob=0.85` and `[11.0-11.5] prob=0.72` (score break at 11.0s)
  - `-mt 1.0` → one range: `[10.0-11.5]` best\_prob=0.85 (all merged by timestamp continuity)

  When used with `--top` or `--top-per-video`, the limit counts ranges (not individual frames).

#### AI Agent / Programmatic Output (`--json`)

Use `--json` (or `-j`) to output results as a clean JSON array instead of ASCII table:

```bash
# Get top 3 matching frames as JSON
scs search video.mp4 --q "activity=yes:prob>=0.8" --top 3 --json
```

**Output format (individual frames):**
```json
[
  {
    "range": false,
    "timestamp_seconds": 1528.861,
    "frame_index": 45820,
    "question_name": "act_category",
    "top_choice": "oral_on_female",
    "top_probability": 0.767853,
    "confidence": 0.91,
    "video_id": 68,
    "run_id": 68
  }
]
```

**Output format (merged ranges with `-mst`/`-mt`):**
```json
[
  {
    "range": true,
    "start_timestamp_seconds": 1528.861,
    "end_timestamp_seconds": 1529.828,
    "duration_seconds": 0.968,
    "frame_count": 2,
    "question_name": "act_category",
    "top_choice": "oral_on_female",
    "top_probability": 0.767853,
    "confidence": null,
    "frame_indices": [45820, 45849],
    "video_id": 68,
    "run_id": 68
  }
]
```

This is designed for easy consumption by LLM agents, CI pipelines, or any tool that needs structured data.

- `--save-frame DIR` (`-sf`): Save extracted frames to directory (JPG only)
- `--model MODEL` (`-m`): Filter results by decision model name (e.g., clef-flash:9b, d1:3b, perplexity/pplx-decider-v1.3)
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
| `--json` | `-j` | — | Output JSON array for AI agents / programmatic use (overrides table) |
| `--save-frame` | `-sf` | — | Extract frame images |
| `--model` | `-m` | — | Filter by decision model name |
| `--run-id` | `-ri` | — | Filter by run ID |
| `--cache` | `-c` | — | Cache path |
| `--stitch` | `-st` | — | Video stitching |

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
| `model` | TEXT | Decision model name (e.g., `clef-flash:9b`, `perplexity/pplx-decider-v1.3`) |
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
| `confidence` | REAL | Model confidence/certainty (0.0–1.0) |
| `raw_response` | TEXT | Raw decision model response JSON |

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

### Isolated Frame Time Padding (`isolated_frame_time`)

When stitching, frames are grouped into sections based on timestamp proximity. By default, any gap larger than `--merge-same-threshold` (or 1 sec if not set) creates a new section.

The `isolated_frame_time` setting adds **extra padding seconds** around each group's end timestamp before checking for isolation. This helps bridge small gaps where you want related content stitched together even though they're slightly beyond the merge threshold.

**How it works:**
- If `isolated_frame_time = 3` and `threshold = 1`: maximum allowed gap between groups is 4 seconds
- Gap ≤ 1 second → direct merge (clearly related)
- Gap 1–4 seconds → pad and merge (possibly related, keep together)
- Gap > 4 seconds → isolate (unrelated)

**Configuration:** Set in `config.toml`:
```toml
isolated_frame_time = 3   # Seconds of padding after isolated groups (default: 3)
merge_same_threshold = 2  # Max gap for direct grouping (optional)
```

With both set above, the stitching logic allows merging groups with gaps up to 5 seconds (2 + 3). Setting `isolated_frame_time = 0` disables padding — only exact threshold matching applies.

## isearch — Experimental Image-Based Search

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
- `--model MODEL`: Filter results by decision model name (e.g., clef-flash:9b, d1:3b, perplexity/pplx-decider-v1.3)
- `--run-id RUN_ID`: Filter results to a specific analysis run ID
- `--save-frame DIR`: Directory to save extracted frames (JPG only)
- `--stitch [PATH]`: Stitch matched frames into a single video using ffmpeg. Uses `--merge-result-threshold` (default 1 sec) to group consecutive frames with same score into ranges before stitching
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

## Query Subcommand (`scs query`)

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
