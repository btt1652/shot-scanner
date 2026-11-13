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
cache.npz + query (text/image) → Load embeddings → Encode query → Similarity search
    → Top-k ranking → Optional frame extraction → Optional Ollama reranking
```
- Loads cached embeddings and model
- Encodes query (text/image) using same CLIP model
- Computes similarities, applies thresholds (`raw_floor`, `norm_threshold`)
- Ranks by similarity (highest first)
- **Frame extraction**: For each ranked result, extracts frame from cached `video_path`
- **Reranking**: If `--decision_model_rerank` enabled, checks each frame with Ollama decision model
  - Each frame is reranked independently using a fused score of CLIP similarity and Ollama's judgment
  - Results whose fused score is below `rerank_threshold` are discarded
  - Passing results are retained with their original timestamps and scores
  - Configurable via `rerank_threshold` in `config.toml` or `--decision-model-rerank` CLI flag

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
- `md5` - MD5 hash of the source video for integrity validation at search time

### `[search]`
Search-mode defaults (used when running `vsearch search`):
- `top_k` - Maximum number of top results to return (-1 = all, 0 = none)
- `raw_floor` - Absolute noise floor for raw similarity
- `norm_threshold` - Normalized score threshold
- `json_output` - Output results in JSON format
- `save_frames` - Directory to save matched frame images
- `video` - Path to original video file
- `decision_model_rerank` - Enable Ollama decision model reranking
- `ollama_endpoint` - Ollama API endpoint (first priority). Accepts a single URL or a list of endpoints tried in order until one is reachable. Bare hosts are accepted too, e.g. `192.168.1.190` or `localhost:11434`.
- `ollama_decision_model` - Ollama decision model name, or a list of model names tried in order (primary first, fallback second) if a model call fails
- `rerank_threshold` - Threshold for the fused reranking score
- `rerank_weight_clip` - Weight given to the CLIP similarity score (0.0 to 1.0) during reranking
- `rerank_weight_ollama` - Weight given to the Ollama judgment score (0.0 to 1.0) during reranking

**CLI flags always override config values when specified.** If a flag is not given, the config value is used. If neither is set, the built-in default applies.

Example `config.toml`:
```toml
[index]
output_dir = "."
model = "openai/clip-vit-base-patch32"
sample_rate = 1

[search]
top_k = -1
raw_floor = 0.25
norm_threshold = 0.75
json_output = false
save_frames = ""
video = ""
decision_model_rerank = false
ollama_endpoint = ["192.168.1.190", "localhost"]
ollama_decision_model = ["d3:4b", "clef-flash"]
rerank_threshold = 0.5
rerank_weight_clip = 0.5
rerank_weight_ollama = 0.5
```

## Ollama endpoint resolution

When `decision_model_rerank` is enabled, the Ollama endpoint is resolved at search time:

1. Each endpoint in `ollama_endpoint` is tried in order (primary first).
2. The first reachable endpoint is used for validation.
3. If none is reachable, the first endpoint in the list is used anyway (validation requests will fail per-frame and be discarded).

This lets you configure a remote server as primary and a local fallback in a single key, e.g.:

```toml
ollama_endpoint = ["192.168.1.190", "localhost"]
```

Bare hosts (with or without a port) are normalized to `http://<host>:11434/v1/systemone`. The same list can be passed on the CLI with `--ollama-endpoint 192.168.1.190,localhost`.

The decision model (`ollama_decision_model`) also accepts a list and works the same way at validation time: each frame is sent to the first model; if that model call fails (e.g. model not loaded on the server), the next model is tried for that frame before the frame is discarded. Pass a fallback list on the CLI with `--ollama-model d3:4b,clef-flash`.

## Reranking Logic

When `decision_model_rerank` is enabled, the system:
1. Retrieves the top `K` results using CLIP.
2. Queries Ollama for a validation score for each of those `K` results.
3. Calculates a **fused score**: `(CLIP_score * rerank_weight_clip) + (Ollama_score * rerank_weight_ollama)`.
4. Reranks the top `K` results based on this fused score.
5. Filters out any results where the fused score is below `rerank_threshold`.

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
- `--model MODEL` - CLIP model variant (default: openai/clip-vit-base-patch32)
- `--verbose, -v` - Enable verbose progress output

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
- `--raw-floor` - Absolute noise floor for raw similarity (default: 0.25)
- `--norm-threshold` - Normalized score threshold (default: 0.75)
- `--verbose, -v` - Enable verbose progress output
- `--save-frames, -f` - Directory to save matched frame images as `{cache_name}_{seconds_milliseconds}_{score}_{norm_score}.jpg` (e.g., `{cache_name}_252.300_{score}_{norm_score}.jpg`)
- `--video` - Path to original video file (required with --save-frames if not recorded in cache, or if video was moved)
- `--decision-model-rerank` - Use local Ollama decision model to rerank top-K selected frames. Each frame is checked independently; results whose fused score is below `rerank_threshold` are discarded from match results.
- `--rerank_threshold` - Threshold for the fused reranking score (default: 0.5). Results with score < threshold are discarded.
- `--ollama_endpoint` - Ollama API endpoint (default: http://localhost:11434/v1/systemone)
- `--ollama_model` - Ollama model name for decision validation. Config key: `ollama_decision_model`.

**Auto video path:** If `--save_frames` or `--decision_model_rerank` is used, vsearch looks for `video_path` stored in the `.npz` cache from the indexing step. If the cached video path exists, it is used automatically — you don't need to pass `--video`. If the video was moved and the cached path no longer exists, vsearch exits with an error and you must provide `--video` with the new path.

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
