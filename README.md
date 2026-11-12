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

5. Save search results to a text file:
   ```
   vsearch search cache.npz -t "a person running" -s /tmp/results
   ```

6. Save matched frames as images (video path is auto-loaded from cache):
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
- Saves: `embeddings`, `timestamps`, `model`, `video_path` to cache

### Search mode
```
cache.npz + query (text/image) → Load embeddings → Encode query → Similarity search
    → Top-k ranking → Optional frame extraction → Optional Ollama validation
```
- Loads cached embeddings and model
- Encodes query (text/image) using same CLIP model
- Computes similarities, applies thresholds (`raw_floor`, `norm_threshold`)
- Ranks by similarity (highest first)
- **Frame extraction**: For each ranked result, extracts frame from cached `video_path`
- **Validation**: If `--decision-model-validate` enabled, checks each frame with Ollama decision model
  - Results fail validation (score < `decision_model_threshold`) are discarded
  - Validation stops at first failure — all lower-score results are discarded
  - Configurable via `decision_model_threshold` in `config.toml` or `--decision-model-threshold` CLI flag

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
        "similarity_score": 0.3125,
        "normalized_score": 1.0
      }
    ]
  }
  ```
- **Files**: Optional `--save-results` (text) and `--save-frames` (images) options

### Config precedence
1. CLI flags override config.toml values
2. Config.toml values override defaults
3. Saved config values are merged with user-provided config values on load


## Config File

vsearch uses a configuration file at `config.toml` in the repo root to store default values. Settings are grouped into three sections:

### `[global]`
Shared defaults applied to both modes.

### `[index]`
Index-mode defaults (used when running `vsearch index`):
- `output_dir` - Default output directory for `.npz` cache
- `model` - CLIP model variant
- `sample_rate` - Frame sampling interval in seconds
- `verbose` - Enable verbose progress output

### `[search]`
Search-mode defaults (used when running `vsearch search`):
- `top_k` - Maximum number of top results to return (-1 = all)
- `raw_floor` - Absolute noise floor for raw similarity
- `norm_threshold` - Normalized score threshold
- `json_output` - Output results in JSON format
- `save_results` - Path to save results text file
- `save_frames` - Directory to save matched frame images
- `video` - Path to original video file
- `validate` - Use local Ollama to validate results
- `ollama_endpoint` - Ollama API endpoint

**CLI flags always override config values when specified.** If a flag is not given, the config value is used. If neither is set, the built-in default applies.

Example `config.toml`:
```toml
[global]
verbose = false

[index]
output_dir = "."
model = "openai/clip-vit-base-patch32"
sample_rate = 1
verbose = false

[search]
verbose = false
top_k = -1
raw_floor = 0.25
norm_threshold = 0.75
json_output = false
save_results = null
save_frames = null
video = null
validate = false
ollama_endpoint = "http://localhost:11434/v1/systemone"
```

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
- `-o, --output OUTPUT` - Path to save the output cache archive (.npz) (default: current directory). If a directory, auto-generates filename: `{video_name}_{model_name}_{sample_rate}.npz`
- `-r, --sample-rate SAMPLE_RATE` - Frame sampling interval in seconds (default: 1)
- `--model MODEL` - CLIP model variant (default: openai/clip-vit-base-patch32)
- `--verbose, -v` - Enable verbose progress output

The `.npz` cache stores:
- `embeddings` - CLIP feature vectors per keyframe
- `timestamps` - Frame timestamps in seconds
- `model` - CLIP model used for indexing
- `video_path` - Path to video file at index time (used for auto frame extraction)

### `search`

Query a pre-computed cache using either a text prompt or a reference image path.

Flags:
- `cache` - Path to pre-computed embeddings cache (.npz) (required)
- `-t, --text` - Natural language search prompt
- `-i, --image` - Path to reference image file for visual similarity matching
- `-k, --top-k` - Maximum number of top results to return (default: all). Use `0` for no results, `-1` for unlimited.
- `--json` - Output results in strict JSON format to stdout for AI agent consumption
- `--raw-floor` - Absolute noise floor for raw similarity (default: 0.25)
- `--norm-threshold` - Normalized score threshold (default: 0.75)
- `--verbose, -v` - Enable verbose progress output
- `--save-results, -s` - Save results to text file. If directory, auto-generates `{cache_name}_search_results_YYYYMMDD_HHMMSS.txt`
- `--save-frames, -f` - Directory to save matched frame images as `{cache_name}_{HHMMSS}_{score}_{norm_score}.jpg`
- `--video` - Path to original video file (required with --save-frames if not recorded in cache, or if video was moved)
- `--decision-model-validate` - Use local Ollama decision model to validate top-K selected frames. Results that fail validation are discarded from match results.
- `--decision-model-threshold` - Ollama decision model threshold (default: 0.5). Results with score < threshold are discarded.
- `--ollama-endpoint` - Ollama API endpoint (default: http://localhost:11434/v1/systemone)
- `--ollama-model` - Ollama model name for decision validation (default: clef-flash). Config key: `ollama_decision_model`.

**Auto video path:** If `--save-frames` or `--decision-model-validate` is used, vsearch looks for `video_path` stored in the `.npz` cache from the indexing step. If the cached video path exists, it is used automatically — you don't need to pass `--video`. If the video was moved and the cached path no longer exists, vsearch exits with an error and you must provide `--video` with the new path.

## Expected JSON Output Schema (`--json` flag)

When `--json` is specified, the application prints strictly valid JSON to stdout matching this schema:

```json
{
  "query": "string (text prompt or image file path)",
  "results": [
    {
      "timestamp_seconds": 252.0,
      "timestamp_formatted": "04:12",
      "similarity_score": 0.3125,
      "normalized_score": 1.0
    }
  ]
}
```

## Architecture

The tool follows a two-phase lifecycle:

1. **Indexing phase** (`vsearch index`): Extracts frames at the specified sample rate, encodes each frame with CLIP, L2-normalizes the feature vectors, and saves them into a compressed `.npz` archive alongside the original timestamps, model ID, and video path.

2. **Query phase** (`vsearch search`): Loads the cached embeddings, auto-detects the CLIP model from the cache, encodes the query (text or image) with the same CLIP model, computes cosine similarity via dot product, applies min-max normalization and dual-threshold filtering (normalized >= threshold AND raw >= floor), sorts by score descending, and returns the top-k matching timestamps.

## Legacy Cache Compatibility

`.npz` caches created before the `model` field was added do not contain model metadata. Running `vsearch search` on such caches will fail with an error message instructing you to re-index the video. This prevents silently producing incorrect results due to model mismatch.

## Target Environment

Optimized for CPU execution (compatible with standard multi-core desktop CPUs, 16GB RAM) with optional CUDA support if a GPU is present.
