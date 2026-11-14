# vsearch - Semantic Video Timestamp CLI

A lightweight, fully offline command-line interface (CLI) tool designed to extract keyframes from local video files, encode them into vector embeddings using an open-source Vision-Language Model (CLIP), and cache them locally. Users or AI agents can then query the cached video instantly via text prompts or reference images to retrieve matching exact timestamps.

## Quick Start

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Index a video file (video path is first positional arg):
   ```
   vsearch index myvideo.mp4 -o cache.npz
   ```

3. Search the cache with text (model is auto-detected from cache, no `--model` needed):
   ```
   vsearch search cache.npz -t "a person running"
   ```

4. Search with reference image:
   ```
   vsearch search cache.npz -i reference.jpg
   ```

5. Save matched frames as images (video path is auto-loaded from cache):
   ```
   vsearch search cache.npz -t "a person running" -f /tmp/frames
   ```

## Program Flow

### Index mode
```
video_path → Extract frames (OpenCV) → Encode images to embeddings → Save to .npz
```
- Reads video, extracts frames based on `sample_rate`
- Encodes each frame using CLIP model
- Saves: `embeddings`, `timestamps`, `model`, `video_path`, `md5` to cache

### Search mode
```
cache.npz + query (text/image) → Load embeddings → Encode query → Raw cosine similarity
    → Sort descending → raw_floor / percentile filter → Top-k cap
    → Optional frame extraction → Optional Ollama fusion → Output
```
- Loads cached embeddings and model
- Encodes query (text/image) using same CLIP model
- Computes **raw cosine similarity** (dot product of L2-normalized vectors; `[0, 1]` for CLIP)
- Applies the calibrated floor (`raw_floor`) and/or optional percentile cutoff, then the `top_k` cap
- Ranks by raw similarity (highest first)
- **Frame extraction**: For each ranked result, extracts frame from cached `video_path`
- **Fusion**: If `--decision-model-fuse` enabled, checks each remaining frame with the Ollama decision model **after** the CLIP ranking, filtering and top-k cap:
  - Each frame is scored by Ollama and fused with its CLIP cosine similarity
  - Fused score: `(cosine_similarity * fuse_weight_clip) + (ollama_score * fuse_weight_ollama)`
  - Results are re-sorted by fused score; results whose fused score is below `fuse_threshold` are discarded
  - Passing results are returned with their original timestamps and scores
  - Configurable via `fuse_threshold` in `config.toml` or `--decision-model-fuse` CLI flag

### Output formats
- **Default**: Human-readable table to stdout
- **JSON**: Structured JSON matching schema:
  ```json
  {
    "query": "search prompt",
    "results": [
      {
        "timestamp_seconds": 252.0,
        "timestamp_formatted": "04:12",
        "score": 0.5123
      }
    ]
  }
  ```
- **Files**: Optional `--save-frames` (images) options

### Config precedence
1. CLI flags override config.toml values
2. Config.toml values override defaults
3. Saved config values are merged with user-provided config values on load


## Config File

vsearch uses a configuration file at `config.toml` in the repo root to store default values. Settings are grouped into two sections:

### `[index]`
Index-mode defaults (used when running `vsearch index`):
- `output_dir` - Default output directory for `.npz` cache
- `model` - CLIP model variant
- `sample_rate` - Frame sampling interval in seconds
- `batch_size` - Number of frames to encode in a single CLIP forward pass during indexing (default: 32). Larger batches reduce Python overhead and can speed up indexing on multi-core CPUs, at the cost of higher peak memory usage.

### `[search]`
Search-mode defaults (used when running `vsearch search`):
- `top_k` - Maximum number of top results to return (-1 = all, 0 = none)
- `raw_floor` - Absolute calibrated cosine similarity floor (e.g. 0.25 means 25% similarity). Results below this are discarded.
- `json_output` - Output results in JSON format
- `save_frames` - Directory to save matched frame images
- `video` - Path to original video file
- `decision_model_fuse` - Enable Ollama decision model fusion (true/false)
- `ollama_host` - Ollama API endpoint (single string, e.g. `"192.168.1.190"`). If unreachable, automatically falls back to `localhost`. Fallback state is remembered for the current run.
- `ollama_decision_model` - Ollama decision model name, or a list of model names tried in order (primary first, fallback second) if a model call fails
- `ollama_frame_scale` - Scale factor for frames sent to Ollama decision model (0.0-1.0, default: 0.2). Lower values = faster inference on CPU
- `fuse_threshold` - Threshold for the fused score (default: 0.5). Results with score < threshold are discarded.
- `fuse_weight_clip` - Weight given to the CLIP similarity score (0.0 to 1.0) during fusion
- `fuse_weight_ollama` - Weight given to the Ollama judgment score (0.0 to 1.0) during fusion
- `percentile` - Percentile cutoff (0.0 to 1.0, e.g. 0.2 keeps top 20%). Setting this automatically enables percentile filtering.

**CLI flags always override config values when specified.** If a flag is not given, the config value is used. If neither is set, the built-in default applies.

Example `config.toml`:
```toml
[index]
output_dir = "."
model = "openai/clip-vit-base-patch32"
sample_rate = 1
batch_size = 32

[search]
top_k = -1
raw_floor = 0.25
json_output = false
save_frames = ""
video = ""
decision_model_fuse = false
ollama_host = "192.168.1.190"
ollama_frame_scale = 0.2
ollama_decision_model = ["d3:4b", "clef-flash"]
fuse_threshold = 0.5
fuse_weight_clip = 0.3
fuse_weight_ollama = 0.7
percentile = 0.1
```

## Ollama endpoint resolution

When `decision_model_fuse` is enabled, the Ollama endpoint is resolved at search time:

1. The primary endpoint in `ollama_host` is tried first.
2. If reachable, it is used and the result is cached for the current run.
3. If unreachable, automatically falls back to `localhost`. The fallback state is remembered for the current run (no re-checking).
4. If both are unreachable, the primary endpoint is used anyway (validation requests will fail per-frame and be discarded).

This lets you configure a remote server as primary with automatic local fallback:

```toml
ollama_host = "192.168.1.190"
```

Bare hosts (with or without a port) are normalized to `http://<host>:11434/v1/systemone`. The same can be passed on the CLI with `--ollama-host 192.168.1.190`.

### Model fallback

The decision model (`ollama_decision_model`) also accepts a list and works the same way at validation time: each frame is sent to the first model; if that model call fails with a definitive error (e.g. model not found / 404), the next model is tried for that frame. **Timeouts are not treated as permanent failures** - the model will be retried on the next frame. Once a model succeeds, it is remembered as the resolved model for all subsequent frames in the run.

Pass a fallback list on the CLI with `--ollama-model d3:4b,clef-flash`.

## Scoring & Thresholding

### Raw cosine similarity ranking

vsearch ranks frames by **raw cosine similarity** (dot product of L2-normalized embeddings). Scores are absolute and comparable across videos (e.g. 0.75 means 75% cosine similarity regardless of which video you searched).

Two filtering modes are available:

**1. Calibrated floor (`raw_floor`) — recommended**
Set a minimum similarity threshold (e.g. `--raw-floor 0.5`). Results below this are discarded. Since results are sorted descending, ranking stops early once similarity falls below this floor.

**2. Percentile-based filtering**
Keep only the top N% of frames by similarity, regardless of absolute score. Useful when you don't know what cosine similarity values to expect. Example: `--percentile 0.2` keeps only the top 20%.

You can combine both: `--raw-floor 0.5 --percentile 0.2` keeps up to the top 20% but only if similarity is at least 0.5.

### Top-k vs percentile

These two controls are different, and are applied **after** `raw_floor`:

| | `--top-k` | `--percentile X` |
|---|---|---|
| What it limits | A fixed **count** of results | A **fraction** of all indexed frames |
| Known in advance? | Yes — you choose the exact number | No — depends on how many frames were indexed |
| Example | `--top-k 10` → at most 10 results | `--percentile 0.2` → keep top 20% of all frames |
| `-1` | No cap (default) | — |
| `0` | No results | — |

Order of operations for a search:
1. Compute raw cosine similarity for every frame.
2. Sort descending.
3. Apply `raw_floor` and (if enabled) the percentile cutoff.
4. Apply the `top_k` cap.
5. (Optional) fuse the remaining frames with Ollama, then re-sort and apply `fuse_threshold`.

So `top_k` is a **fixed-count cap** of the highest-scoring frames that already passed the other filters, while `percentile` filters by **share of the total frame set** (whose count you don't know up front).

### Fusing (Ollama decision model)

Fusion runs **last** — after the CLIP raw-cosine ranking, `raw_floor`/percentile filtering and the `top_k` cap. When `decision_model_fuse` is enabled, the system:
1. Takes the frames remaining after step 4 above (CLIP-ranked, filtered, top-k capped).
2. Extracts each frame and queries Ollama for a validation score (`ollama_score`, `[0, 1]`).
3. Calculates a **fused score**: `(cosine_similarity * fuse_weight_clip) + (ollama_score * fuse_weight_ollama)`.
   - The CLIP term is the frame's **raw cosine similarity** (the same value reported as `score`).
4. Re-sorts these frames by the fused score (descending).
5. Discards any results where the fused score is below `fuse_threshold`.

This ensures that results that are visually strong but perhaps slightly less "certain" from the LLM still rank highly, while very high-confidence LLM matches move to the top.

## Model Variants

The tool supports multiple CLIP model variants via the `--model` flag during **indexing**:

- `openai/clip-vit-base-patch32` (default) - Good balance of speed and accuracy
- `openai/clip-vit-base-patch16` - Smaller, faster on CPU (~400MB)
- `openai/clip-vit-large-patch14` - Larger, more accurate (~1.3GB, needs more memory)

Example:
```
vsearch index myvideo.mp4 -o cache.npz --model openai/clip-vit-base-patch16
```

**Important:** During searching, `--model` is **not** accepted. The model is auto-detected from the `.npz` cache file to ensure the query is encoded with the same model used during indexing. This prevents mismatched results.

## Commands

### `index`

Process a local video file, extract keyframes, compute embeddings, and save a compressed archive.

Flags:
- `video` - Path to input video file (required)
- `-o, --output OUTPUT` - Path to save the output cache archive (.npz). If a directory, auto-generates filename: `{video_name}_{model_name}_{sample_rate}_{md5_prefix8}.npz`
- `-r, --sample-rate SAMPLE_RATE` - Frame sampling interval in seconds (default: 1)
- `-bs, --batch-size BATCH_SIZE` - Number of frames to encode in a single CLIP forward pass during indexing (default: 32). Larger batches reduce Python overhead and can speed up indexing on multi-core CPUs, at the cost of higher peak memory usage.
- `--model MODEL` - CLIP model variant (default: openai/clip-vit-base-patch32)

Indexing always prints progress logs to stderr.

The `.npz` cache stores:
- `embeddings` - CLIP feature vectors per keyframe
- `timestamps` - Frame timestamps in seconds
- `model` - CLIP model used for indexing
- `video_path` - Path to video file at index time (used for auto frame extraction)
- `md5` - MD5 hash of the source video for integrity validation at search time

### `search`

Query a pre-computed cache using either a text prompt or a reference image path.

Flags:
- `cache` - Path to pre-computed embeddings cache (.npz) (required)
- `-t, --text` - Natural language search prompt
- `-i, --image` - Path to reference image file for visual similarity matching
- `-k, --top-k` - Maximum number of top results to return (default: all). Use `0` for no results, `-1` for unlimited.
- `--json` - Output results in strict JSON format to stdout for AI agent consumption
- `--raw-floor` - Absolute calibrated cosine similarity floor (default: 0.25). Results below this similarity are discarded.
- `--percentile` - Percentile cutoff (0.0 to 1.0, e.g. 0.2 keeps top 20%). Setting this enables percentile filtering automatically.
- `--save-frames, -f` - Directory to save matched frame images as `{cache_name}_{HHMMSS}_{score}.jpg`
- `--video` - Path to original video file (required with --save-frames if not recorded in cache, or if video was moved)
- `--decision-model-fuse` - Use local Ollama decision model to fuse top-K selected frames. Each frame is checked independently; results whose fused score is below `fuse_threshold` are discarded from match results.
- `--fuse-threshold` - Threshold for the fused score (default: 0.5). Results with score < threshold are discarded.
- `--fuse-weight-clip` - Weight given to the CLIP similarity score in the fused score (default: 0.3). CLI flag; also configurable via `fuse_weight_clip` in `config.toml`.
- `--fuse-weight-ollama` - Weight given to the Ollama judgment score in the fused score (default: 0.7). CLI flag; also configurable via `fuse_weight_ollama` in `config.toml`.
- `--ollama-host` - Ollama server addr (default: 192.168.1.190). Falls back to localhost if unreachable. Config key: `ollama_host`.
- `--ollama-frame-scale` - Scale factor for frames sent to Ollama decision model (0.0-1.0, default: 0.2). Lower values = faster inference on CPU. Config key: `ollama_frame_scale`.
- `--ollama-model` - Ollama model name for decision validation. Config key: `ollama_decision_model`.

**Auto video path:** If `--save-frames` or `--decision-model-fuse` is used, vsearch looks for `video_path` stored in the `.npz` cache from the indexing step. If the cached video path exists, it is used automatically — you don't need to pass `--video`. If the video was moved and the cached path no longer exists, vsearch exits with an error and you must provide `--video` with the new path.

**MD5 integrity check:** If `--video` is provided, vsearch verifies its MD5 hash against the `md5` field stored in the cache. If they do not match, vsearch exits with an error.

## Expected JSON Output Schema (`--json` flag)

When `--json` is specified, the application prints strictly valid JSON to stdout matching this schema:

```json
{
  "query": "string (text prompt or image file path)",
  "results": [
    {
      "timestamp_seconds": 252.0,
      "timestamp_formatted": "04:12",
      "score": 0.5123
    }
  ]
}
```

`score` is the raw cosine similarity (0.0–1.0). Min‑max normalization was removed entirely.

## Architecture

The tool follows a two-phase lifecycle:

1. **Indexing phase** (`vsearch index`): Extracts frames at the specified sample rate, encodes them in batches with CLIP (batch size configurable via `--batch-size` or `config.toml`), L2-normalizes the feature vectors, and saves them into a compressed `.npz` archive alongside the original timestamps, model ID, and video path.

2. **Query phase** (`vsearch search`): Loads the cached embeddings, auto-detects the CLIP model from the cache, encodes the query (text or image) with the same CLIP model, computes cosine similarity via dot product, applies calibrated floor filtering (`raw_floor`) or optional percentile cutoff (`percentile`), sorts by score descending, and returns the top-k matching timestamps.
