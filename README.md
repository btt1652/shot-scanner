# vsearch - Decision Model Video Search

A lightweight command-line interface that searches local video files using an Ollama decision model as the sole judge. No CLIP embeddings or pre-built `.npz` caches are required — frames are extracted and validated on the fly.

## Quick Start

1. Install dependencies:

   ```
   pip install -r requirements.txt
   ```

2. Search a video with a text prompt:

   ```
   vsearch search --video myvideo.mp4 -t "a person running"
   ```

3. Limit the number of results:

   ```
   vsearch search --video myvideo.mp4 -t "a person running" -k 5
   ```

4. Save matched frames as JPEG images:

   ```
   vsearch search --video myvideo.mp4 -t "a person running" -sf /tmp/frames
   ```

5. Output JSON:

   ```
   vsearch search --video myvideo.mp4 -t "a person running" --json
   ```

## Decision-Model-Only Mode

`vsearch search` uses the Ollama decision model as the sole judge. It does **not** use CLIP embeddings, `.npz` caches, or fused scores.

### Workflow

1. Frames are sampled from the video at `--decision-model-sample-rate` intervals.
2. Each frame is resized (`--ollama-frame-scale`) and sent to the Ollama server.
3. Frames whose score is below `--decision-model-threshold` are discarded.
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
| `--video` | `-vid` | Path to a video file, directory, or glob pattern (required) |
| `--text` | `-t` | Natural-language search prompt |
| `--image` | `-i` | Path to a reference image file (used as the query string) |
| `--top-k` | `-k` | Maximum number of results to return (default: all passing) |
| `--global-k` | `-gk` | Global top-K across all videos, overriding `--top-k` |
| `--json` | | Output results as strict JSON |
| `--save-frames` | `-sf` | Directory to save matched frame JPEGs |
| `--decision-model-threshold` | `-dmt` | Ollama score threshold (default: 0.5) |
| `--decision-model-sample-rate` | `-dmsr` | Frame sampling interval in seconds (default: 1) |
| `--ollama-workers` | `-ow` | Concurrent Ollama validation threads (default: 1) |
| `--ollama-candidates` | `-oc` | Maximum candidate frames to validate (default: unlimited) |
| `--ollama-host` | `-oh` | Ollama server address (default: `192.168.1.190`). Falls back to `localhost` if unreachable |
| `--ollama-frame-scale` | `-ofs` | Scale factor for frames sent to Ollama (0.0–1.0, default: 0.2). Lower = faster |
| `--ollama-model` | `-dm` | Ollama model name(s), comma-separated fallback list |
| `--recursive` | `-R` | Recursively scan directories for video files |

## Configuration (`config.toml`)

The config file contains a single `[search]` section:

```toml
[search]
top_k = -1
global_k = null
raw_floor = 0.25
json_output = false
save_frames = ""
video = ""
decision_model_fuse = false
decision_model_only = false
decision_model_threshold = 0.5
decision_model_sample_rate = 1
ollama_host = "192.168.1.190"
ollama_frame_scale = 0.2
ollama_decision_model = ["d3:9b", "d3:4b"]
fuse_threshold = 0.5
fuse_weight_clip = 0.3
fuse_weight_ollama = 0.7
percentile = 0.1
ollama_workers = 4
ollama_candidates = 0
```

### Key config values

- `top_k` — maximum results returned (default: `-1` = all passing)
- `global_k` — global cap across all videos
- `json_output` — default to JSON output (`true`/`false`)
- `save_frames` — default output directory for saved frames
- `decision_model_threshold` — Ollama score threshold (default: `0.5`)
- `decision_model_sample_rate` — sampling interval in seconds (default: `1`)
- `ollama_host` — Ollama endpoint (falls back to `localhost` if unreachable)
- `ollama_frame_scale` — image scale factor sent to Ollama (default: `0.2`)
- `ollama_decision_model` — model name or fallback list
- `ollama_workers` — concurrent validation threads (default: `1`)
- `ollama_candidates` — max candidate frames to validate

## Performance Tips

- **Scale down images** with `--ollama-frame-scale` (e.g. `0.15`) — lower = faster inference.
- **Parallel validation** with `--ollama-workers 4` (or higher) — each thread handles one frame.
- **Coarser sampling** with `--decision-model-sample-rate 5` — one frame every 5 seconds.
- **Lower the threshold** with `--decision-model-threshold 0.4` to keep more passing frames.

## Architecture

- **No indexing step** — videos are read directly with OpenCV.
- **No `.npz` cache** — frames are extracted and validated on the fly.
- **Ollama-only scoring** — `final_score = Ollama_score`; CLIP is never invoked.
- **Endpoint fallback** — if the configured Ollama host is unreachable, `localhost:11434` is used automatically.
- **Parallel validation** — frames are validated concurrently via `ThreadPoolExecutor`.