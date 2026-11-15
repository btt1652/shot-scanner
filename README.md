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
json_output = false
save_frames = ""
video = ""
decision_model_threshold = 0.5
decision_model_sample_rate = 1
ollama_host = "192.168.1.190"
ollama_frame_scale = 0.2
ollama_decision_model = ["d3:9b", "d3:4b"]
ollama_workers = 1
ollama_candidates = 0
```

### Key config values

- `top_k` — maximum results returned (default: `-1` = all passing)
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
- **Scene-change detection** with `--scene-change` — automatically extracts frames at scene boundaries instead of uniform sampling, often much more efficient.

## Architecture

- **No indexing step** — videos are read directly with OpenCV.
- **No `.npz` cache** — frames are extracted and validated on the fly.
- **Ollama-only scoring** — `final_score = Ollama_score`; CLIP is never invoked.
- **Endpoint fallback** — if the configured Ollama host is unreachable, `localhost:11434` is used automatically.
- **Parallel validation** — frames are validated concurrently via `ThreadPoolExecutor`.

## Scene Change Detection

The `--scene-change` and `--scene-threshold` flags enable automatic scene change detection. Instead of blindly sampling every `decision_model_sample_rate` seconds, the app uses FFmpeg’s scene filter to detect scenes and only validates the key frames at those transitions:

```bash
vsearch search --scene-change --scene-threshold 0.5 --video video.mp4 --text "a person"
```

**How it works:**

1. `ffmpeg` runs `-vf "select='gt(scene,THRESHOLD)',metadata=print"` on the video and parses the scene-change timestamps (`pts_time`) from stderr.
2. For each timestamp, the app extracts that frame (using OpenCV seek) and passes it to Ollama.
3. Only scene boundaries are validated, which can drastically reduce processing time and focus validation on moments that likely contain content changes.

**Performance benefits:**
- Many scenes can be detected with fewer frames than uniform sampling.
- Scene thresholds (`0.0‑1.0`) control sensitivity: `0.4` (default) captures moderate changes; `0.7` requires strong changes.
- If no scene changes are detected (e.g., static videos), the app falls back to keyframe detection.
- Scene detection is performed once and then frames are processed using the same validation pipeline as normal sampling.

**Configuration:**
- `scene_change = false` – disable scene detection (uniform sampling).
- `scene_threshold = 0.4` – scene-change sensitivity threshold.

- **No indexing step** — videos are read directly with OpenCV.
- **No `.npz` cache** — frames are extracted and validated on the fly.
- **Ollama-only scoring** — `final_score = Ollama_score`; CLIP is never invoked.
- **Endpoint fallback** — if the configured Ollama host is unreachable, `localhost:11434` is used automatically.
- **Parallel validation** — frames are validated concurrently via `ThreadPoolExecutor`.