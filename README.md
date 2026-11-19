# vsearch — Semantic Video Frame Classification

A lightweight CLI that classifies video frames using an Ollama **choice-question** model as the sole judge. No CLIP embeddings or pre-built `.npz` caches are required — frames are extracted and validated on the fly.

**New in this version:** Multiple questions per frame. Each question's answer is stored as a separate row in a searchable SQLite cache. Query with SQL instead of CLI filters.

---

## Quick Start

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Classify a video with default questions (from `config.toml`):

   ```bash
   vsearch classify myvideo.mp4
   ```

3. Customize questions via CLI (`--questions` takes a JSON array):

   ```bash
   vsearch classify myvideo.mp4 \
     --questions '[
       {"name": "activity", "instructions": "What activity?", "criteria": {"walking": "Person walking", "running": "Person running"}},
       {"name": "time_of_day", "instructions": "Time of day?", "criteria": {"morning": "Morning", "afternoon": "Afternoon"}}
     ]'
   ```

4. Override other settings:

   ```bash
   vsearch classify myvideo.mp4 --decision-model-sample-rate 5 --ollama-frame-scale 0.15
   ```

5. Query cached answers with SQL:

   ```bash
   vsearch cache sql "SELECT video_path, timestamp_seconds, top_choice FROM frame_answers WHERE question_name='activity' AND top_choice='walking'" --cache-file ./.cache.db
   ```

---

## How It Works

1. Frames are sampled from the video at `--decision-model-sample-rate` intervals.
2. Each frame is resized (`--ollama-frame-scale`) and base64-encoded.
3. Consecutive frames with similarity ≥ `same_scene_threshold` (default 0.80) are grouped into same-scene batches (max 4 frames per batch, constrained by Ollama's multi-image limit) and sent to Ollama as a single multi-image request.
4. Ollama receives **all questions at once** and returns a probability distribution for each question's choice criteria.
5. Each question's answer for each frame is stored as a row in `frame_answers` (one frame = N rows, where N = number of questions).
6. No CLI filtering (`--top-k`, `--top-choice`, `--decision-model-floor`) — store everything, filter via SQL.

---

## CLI Flags (`vsearch classify`)

| Flag | Short | Description |
|------|-------|-------------|
| `--questions` | `-qs` | **JSON array** of questions: `[{"name":..., "instructions":..., "criteria":{...}}, ...]`. Replaces config questions when provided. |
| `--decision-model-sample-rate` | `-dmsr` | Frame sampling interval in seconds (default: 3) |
| `--ollama-host` | `-oh` | Ollama server address (default: `192.168.1.190`) |
| `--ollama-frame-scale` | `-ofs` | Image scale factor (0.0–1.0, default: 0.1). Lower = faster. |
| `--ollama-model` | `-dm` | Ollama model name (default: `typesafe/jev-1.14`) |
| `--ollama-timeout` | `-ot` | Ollama request timeout in seconds (default: 60) |
| `--still-frame-threshold` | | Similarity (0–1) above which a frame is considered still (default: 0.95) |
| `--same-scene-threshold` | | Similarity (0–1) above which frames are same scene (default: 0.80) |
| `--recursive` | `-R` | Recursively scan directories for video files |
| `--cache-file` | | Path to SQLite cache file (lookup if present, otherwise save) |
| `--ollama-state` | `-s` | State context string for Ollama (default: `sample frame from a video`) |

**Removed flags** (use SQL instead):
- `--top-k`, `--global-k` → `ORDER BY top_probability DESC LIMIT N` in SQL
- `--top-choice` → `WHERE top_choice = '...'` in SQL
- `--decision-model-floor` → `WHERE top_probability >= 0.5` in SQL
- `--merge-range` → implement in SQL if needed
- `--json` → results are in SQLite, query with SQL

---

## Cache Subcommand (`vsearch cache`)

```bash
# List all cached videos
vsearch cache list --cache-file ./.cache.db

# Show cache statistics
vsearch cache stats --cache-file ./.cache.db

# Prune entries older than N days (default 30)
vsearch cache prune 7 --cache-file ./.cache.db

# Execute custom SQL query
vsearch cache sql "SELECT * FROM frame_answers WHERE question_name='activity'" --cache-file ./.cache.db
```

---

## Configuration (`config.toml`)

The config file uses a `[[classify.questions]]` array-of-tables for multi-question definitions:

```toml
[classify]
decision_model_sample_rate = 3
ollama_host = "192.168.1.190"
ollama_frame_scale = 0.1
ollama_decision_model = "typesafe/jev-1.14"
ollama_timeout = 60
cache_file = "./.cache.db"
still_frame_threshold = 0.95
same_scene_threshold = 0.80
ollama_state_template = "sample frame from a video"

[[classify.questions]]
name = "activity"
instructions = "What activity is happening in this frame?"
criteria = { walking = "Person walking", running = "Person running", sitting = "Person sitting" }

[[classify.questions]]
name = "time_of_day"
instructions = "What time of day is it?"
criteria = { morning = "Morning light", afternoon = "Afternoon light", evening = "Evening light" }

# Example: yes/no (search-equivalent) question
# [[classify.questions]]
# name = "is_valid"
# instructions = "Does the image match the query?"
# criteria = { yes = "the frame matches the query", no = "the frame does not match" }
```

CLI `--questions` **replaces** the entire `classify.questions` array from config when provided.

---

## Cache Schema

The SQLite database contains three tables:

```sql
videos (
    id, video_path, content_hash, params_hash,
    fps, total_frames, stride, created_at, updated_at, cache_status
)

frame_answers (
    id, video_id, frame_index, timestamp_seconds,
    question_name, question_instructions, criteria_json,
    top_choice, top_probability, probabilities_json,
    batch_id, raw_response, created_at,
    UNIQUE(video_id, frame_index, question_name)
)

cache_stats (key, value, updated_at)
```

Indexes: `idx_frame_answers_video`, `idx_frame_answers_choice`, `idx_frame_answers_qname`, `idx_frame_answers_qname_choice`.

### Cache Key (content + params + questions)

The cache matches on **video content AND all Ollama parameters AND the full questions JSON**:

| Component | Description |
|-----------|-------------|
| `content_hash` | BLAKE2b of first/last 8 MB of video file |
| `params_hash` | BLAKE2b of `sample_rate \| frame_scale \| state \| questions_json \| model` (+ thresholds) |

Changing **any question** (name, instructions, criteria) changes `questions_json` → cache miss → fresh run. This ensures correctness.

---

## Query Examples

```bash
# All frames where activity = walking
vsearch cache sql "
  SELECT v.video_path, fa.timestamp_seconds, fa.top_choice, fa.top_probability
  FROM frame_answers fa JOIN videos v ON fa.video_id=v.id
  WHERE fa.question_name='activity' AND fa.top_choice='walking'
" --cache-file ./.cache.db

# All morning scenes (join on frame_index to combine questions)
vsearch cache sql "
  SELECT v.video_path, fa.timestamp_seconds
  FROM frame_answers fa JOIN videos v ON fa.video_id=v.id
  WHERE fa.question_name='time_of_day' AND fa.top_choice='morning'
" --cache-file ./.cache.db

# Top-5 activity frames by probability
vsearch cache sql "
  SELECT video_id, frame_index, top_choice, top_probability
  FROM (
    SELECT *, ROW_NUMBER() OVER (PARTITION BY question_name ORDER BY top_probability DESC) rn
    FROM frame_answers WHERE question_name='activity'
  ) WHERE rn <= 5
" --cache-file ./.cache.db

# Frames with high 'technical' probability (even if not top choice)
vsearch cache sql "
  SELECT v.video_path, fa.timestamp_seconds,
         json_extract(fa.probabilities_json, '$.technical') AS technical_prob
  FROM frame_answers fa JOIN videos v ON fa.video_id=v.id
  WHERE fa.question_name='activity' AND technical_prob > 0.5
" --cache-file ./.cache.db
```

---

## Performance Tips

- **Scale down images** with `--ollama-frame-scale 0.15` — lower = faster inference.
- **Same-scene batching** — consecutive frames with similarity ≥ `same_scene_threshold` are grouped and sent as a single multi-image request (max 4 frames/batch).
- **Coarser sampling** with `--decision-model-sample-rate 5` — one frame every 5 seconds.
- **Use the cache** — re-running with the same questions is instant on cache hit.

---

## Architecture

- **No indexing step** — videos are read directly with OpenCV.
- **No `.npz` cache** — frames are extracted and validated on the fly.
- **Ollama-only scoring** — all questions sent in one request; each question returns its own probability distribution.
- **Endpoint fallback** — if the configured Ollama host is unreachable, `localhost:11434` is used automatically.
- **Same-scene batching** — frames with high visual similarity within a short window are grouped and sent to Ollama as a single multi-image batch request (max 4 frames per batch).

---

## Yes/No (Search-Equivalent) Classification

To replicate the old `vsearch search` behavior, define a yes/no question as a two-choice criteria:

```toml
[[classify.questions]]
name = "is_valid"
instructions = "Does the image match the query?"
criteria = { yes = "the frame matches the query", no = "the frame does not match" }
```

Or via CLI:

```bash
vsearch classify myvideo.mp4 \
  --questions '[{"name":"is_valid","instructions":"Does this frame match?","criteria":{"yes":"matches","no":"does not match"}}]'
```

Then query:

```bash
vsearch cache sql "SELECT video_path, timestamp_seconds FROM frame_answers WHERE question_name='is_valid' AND top_choice='yes'" --cache-file ./.cache.db
```

---

## AI Agent Usage

For AI agents using vsearch as a tool:

```bash
# Initialize/check cache DB
vsearch cache stats --cache-file /path/to/cache.db

# Classify with caching (uses config.toml questions)
vsearch classify video.mp4 --cache-file /path/to/cache.db

# Query cached answers
vsearch cache sql "SELECT video_path, timestamp_seconds, top_choice FROM frame_answers WHERE question_name='activity' AND top_choice='walking'" --cache-file /path/to/cache.db

# Prune old entries
vsearch cache prune 30 --cache-file /path/to/cache.db  # keep last 30 days only
```

---

## Example Ollama Request (built from config)

```json
{
  "model": "typesafe/jev-1.14",
  "state": "sample frame from a video",
  "images": ["base64_image1", "base64_image2"],
  "questions": {
    "activity": {
      "type": "choice",
      "instructions": "What activity is happening in this frame?",
      "criteria": {
        "walking": "Person walking",
        "running": "Person running",
        "sitting": "Person sitting"
      }
    },
    "time_of_day": {
      "type": "choice",
      "instructions": "What time of day is it?",
      "criteria": {
        "morning": "Morning light",
        "afternoon": "Afternoon light",
        "evening": "Evening light"
      }
    }
  }
}
```

---

## Migration from Old Version

| Old | New |
|-----|-----|
| `--ollama-classify-question`, `--ollama-classify-instructions`, `--ollama-classify-criteria` | `--questions` JSON array or `[[classify.questions]]` in config.toml |
| `--top-choice` | SQL: `WHERE top_choice = '...'` |
| `--decision-model-floor` | SQL: `WHERE top_probability >= 0.5` |
| `--top-k`, `--global-k` | SQL: `ORDER BY top_probability DESC LIMIT N` |
| `--merge-range` | Implement in SQL / post-processing |
| `frames` table | Replaced by `frame_answers` (one row per question per frame) |
| `ollama_classify_*` config keys | Replaced by `[[classify.questions]]` array |