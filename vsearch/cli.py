"""Command-line interface for vsearch."""

import argparse
import glob
import hashlib
import json
import os
import sys
import urllib.request
import base64

import cv2
import numpy as np

from .engine import VideoSearchEngine, BATCH_SIZE
from .config import load_config, get_config_value


DEFAULT_VIDEO_EXTENSIONS = [".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv"]


def _resolve_video_paths(inputs, recursive=False, extensions=None):
    """Resolve a list of inputs (files, directories, glob patterns) into concrete video file paths.

    Args:
        inputs: list of input strings (file paths, directory paths, or glob patterns)
        recursive: if True, scan directories recursively for video files
        extensions: list of extensions (with leading dot) to match in directories; default DEFAULT_VIDEO_EXTENSIONS

    Returns:
        (video_paths, missing) where video_paths is a de-duplicated list of absolute paths
        and missing is a list of inputs that matched nothing.
    """
    if extensions is None:
        extensions = DEFAULT_VIDEO_EXTENSIONS
    exts_lower = tuple(e.lower() for e in extensions)

    resolved = []
    seen = set()
    missing = []

    def _add(path):
        p = os.path.abspath(path)
        if os.path.isfile(p) and p not in seen:
            seen.add(p)
            resolved.append(p)

    for inp in inputs:
        matched_any = False

        # 1. Directory: scan for video files
        if os.path.isdir(inp):
            walker = os.walk(inp) if recursive else [(inp, [], os.listdir(inp))]
            for root, dirs, files in walker:
                for fname in files:
                    if fname.lower().endswith(exts_lower):
                        _add(os.path.join(root, fname))
                        matched_any = True

        # 2. Glob pattern
        elif any(c in inp for c in "*?[]"):
            for m in sorted(glob.glob(inp, recursive=recursive)):
                if os.path.isfile(m):
                    _add(m)
                    matched_any = True

        # 3. Literal file path
        else:
            if os.path.isfile(inp):
                _add(inp)
                matched_any = True

        if not matched_any:
            missing.append(inp)

    return resolved, missing


def cmd_index(args):
    cfg = load_config()
    model = get_config_value(cfg, "index", "model", getattr(args, "model", None))
    sample_rate = get_config_value(cfg, "index", "sample_rate", getattr(args, "sample_rate", None))
    batch_size = get_config_value(cfg, "index", "batch_size", getattr(args, "batch_size", BATCH_SIZE))
    recursive = getattr(args, "recursive", False)
    dry_run = getattr(args, "dry_run", False)
    
    # Parse extensions arg
    ext_arg = getattr(args, "ext", None)
    if ext_arg:
        extensions = [e if e.startswith(".") else "." + e for e in ext_arg.split(",") if e.strip()]
    else:
        extensions = None

    # Print config values (always)
    print(f"Config: model={model}, sample_rate={sample_rate}s, batch_size={batch_size}", file=sys.stderr)

    video_paths, missing = _resolve_video_paths(args.video, recursive=recursive, extensions=extensions)

    if missing:
        for m in missing:
            print(f"Warning: no video matched input '{m}'", file=sys.stderr)

    if not video_paths:
        print("Error: no video files found to index.", file=sys.stderr)
        sys.exit(1)

    # Dry run: just list matched files
    if dry_run:
        print(f"Dry run: {len(video_paths)} video file(s) would be indexed:", file=sys.stderr)
        for i, vp in enumerate(video_paths, 1):
            print(f"  [{i}] {vp}", file=sys.stderr)
        return

    output_path = get_config_value(cfg, "index", "output_dir", getattr(args, "output", None))
    batch_mode = len(video_paths) > 1

    # Validate output path for batch mode
    if batch_mode:
        if output_path and output_path.lower().endswith(".npz"):
            print(
                f"Error: batch indexing of {len(video_paths)} videos requires -o to be a directory, not a .npz file.",
                file=sys.stderr,
            )
            sys.exit(1)
        os.makedirs(output_path, exist_ok=True)
        output_is_dir = True
    else:
        # Single video
        if output_path and output_path.lower().endswith(".npz"):
            output_is_dir = False
        else:
            # Treat as directory (create if needed)
            if output_path:
                os.makedirs(output_path, exist_ok=True)
            output_is_dir = True

    engine = VideoSearchEngine(model_id=model)

    total = len(video_paths)
    failures = []
    for idx, video_path in enumerate(video_paths, 1):
        prefix = f"[{idx}/{total}]" if total > 1 else ""
        print(f"{prefix} Indexing {video_path}...", file=sys.stderr)

        try:
            embeddings, timestamps = engine.extract_and_encode(video_path, sample_rate, batch_size)
        except Exception as e:
            print(f"{prefix} Error encoding {video_path}: {e}", file=sys.stderr)
            failures.append((video_path, str(e)))
            continue

        # Compute MD5 of the source video for integrity checking
        video_md5 = hashlib.md5()
        with open(video_path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                video_md5.update(chunk)
        video_md5_hex = video_md5.hexdigest()

        # Auto-generate filename
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        model_name = model.split("/")[-1]  # e.g., clip-vit-base-patch32
        md5_prefix = video_md5_hex[:8]  # first 8 chars
        out_name = f"{video_name}_{model_name}_{sample_rate}s_{md5_prefix}.npz"

        if output_is_dir:
            out_file = os.path.join(output_path, out_name)
        else:
            out_file = output_path

        np.savez_compressed(
            out_file,
            embeddings=embeddings,
            timestamps=timestamps,
            model=model,
            video_path=video_path,
            md5=video_md5_hex,
        )
        print(f"{prefix} Cache saved to {out_file} ({len(embeddings)} keyframes)", file=sys.stderr)

    # Report failures
    if failures:
        print(f"\nCompleted with {len(failures)} failure(s):", file=sys.stderr)
        for vp, err in failures:
            print(f"  {vp}: {err}", file=sys.stderr)
        sys.exit(1)

    print(f"\nDone: indexed {total} video(s).", file=sys.stderr)


def _format_results_text(results):
    """Format results as human-readable text lines."""
    lines = []
    for r in results:
        lines.append(
            f"ts={r['timestamp_formatted']}  sim={r['score']:.4f}"
        )
    return "\n".join(lines)


def _extract_frame_base64(video_path, timestamp, frame_scale=0.2):
    """Extract a single frame from video at given timestamp and return base64-encoded JPEG.
    
    Args:
        video_path: Path to video file
        timestamp: Frame timestamp in seconds
        frame_scale: Scale factor for frame (0.0-1.0, default: 0.2)
                   Image is resized to this percentage of original size before sending to Ollama.
                   Lower values = faster inference (more tokens = slower)
                   Typical range: 0.1-0.3 (faster inference) or 0.5-1.0 (higher accuracy)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return ""
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 0:
        fps = 30.0
    frame_number = int(timestamp * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return ""
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Downscale frame before sending to Ollama decision model to speed up inference
    # Vision models process pixels as tokens - full-res frames are very slow on CPU
    # Scale by frame_scale percentage of original size while preserving aspect ratio
    h, w = img.shape[:2]
    new_h = int(h * frame_scale)
    new_w = int(w * frame_scale)
    if new_h < 1 or new_w < 1:
        new_h, new_w = max(1, int(h * frame_scale)), max(1, int(w * frame_scale))
    img_small = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    # Encode to JPEG using imencode
    success, encoded = cv2.imencode(".jpg", img_small, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not success:
        return ""
    # Convert to base64
    return base64.b64encode(encoded.tobytes()).decode("utf-8")


def _extract_frame_at(video_path, timestamp, out_path):
    """Extract a single frame at given timestamp and save to out_path as JPEG."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return False
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 0:
        fps = 30.0
    frame_number = int(timestamp * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return False
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    cv2.imwrite(out_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
    return True


def _extract_frames_lazy(results, video_path, verbose=False, frame_scale=0.2):
    """Yield frames one at a time (lazy generator) to avoid holding all in memory."""
    if not video_path:
        return
    for r in results:
        ts = r["timestamp_seconds"]
        ts_str = f"{int(ts)//3600:02d}{int(ts)%3600//60:02d}{int(ts)%60:02d}"
        frame_b64 = _extract_frame_base64(video_path, ts, frame_scale=frame_scale)
        if frame_b64:
            yield frame_b64
        elif verbose:
            print(f"Failed to extract frame at {ts:.1f}s", file=sys.stderr)


def _save_matched_frames(results, npz_name, video_path, output_dir, verbose=False):
    """Extract and save matched frames as JPEG images."""
    os.makedirs(output_dir, exist_ok=True)
    saved = 0
    frame_paths = []
    for r in results:
        ts = r["timestamp_seconds"]
        # Format timestamp as HH:MM:SS (with hours if >= 1 hour)
        if ts >= 3600:
            ts_formatted = f"{int(ts)//3600:02d}:{int(ts)%3600//60:02d}:{int(ts)%60:02d}"
        else:
            ts_formatted = f"{int(ts)//60:02d}:{int(ts)%60:02d}"
        # Use the original video filename (without extension) if provided, else npz_name
        if video_path:
            video_name = os.path.splitext(os.path.basename(video_path))[0]
        else:
            video_name = npz_name
        out_path = os.path.join(output_dir, f"{video_name}_{ts_formatted}_{r['score']:.4f}.jpg")
        if _extract_frame_at(video_path, ts, out_path):
            saved += 1
            frame_paths.append(out_path)
            if verbose:
                print(f"Saved frame at {ts:.1f}s -> {out_path}", file=sys.stderr)
        elif verbose:
            print(f"Failed to extract frame at {ts:.1f}s", file=sys.stderr)
    return saved, frame_paths


def _as_list(value):
    """Normalize a value into a list of non-empty strings. Strips whitespace and splits comma-separated values."""
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        items = list(value)
    else:
        value = str(value).strip()
        if not value:
            return []
        if "," in value:
            items = value.split(",")
        else:
            items = [value]
    return [str(item).strip() for item in items if str(item).strip()]


def _normalize_ollama_endpoint(endpoint):
    """Normalize an Ollama endpoint to a full URL. Accepts bare hosts like
    '192.168.1.190' or 'localhost:11434' or full URLs like 'http://192.168.1.190:11434/v1/systemone'."""
    if not endpoint:
        return None
    ep = str(endpoint).strip().rstrip("/")
    if ep.startswith(("http://", "https://")):
        return ep
    host_part = ep.split("://")[-1]
    if "/" in host_part:
        host_part = host_part.split("/")[0]
    host = host_part
    port = 11434
    if ":" in host:
        host, port_str = host.split(":", 1)
        port = int(port_str) if port_str.isdigit() else 11434
    return f"http://{host}:{port}/v1/systemone"


def _ollama_endpoint_reachable(endpoint, timeout=5.0):
    """Check if the Ollama endpoint is reachable.

    Any HTTP response (including 405 Method Not Allowed) means the server is
    up; only connection errors count as unreachable.
    """
    endpoint = _normalize_ollama_endpoint(endpoint)
    if not endpoint:
        return False
    try:
        req = urllib.request.Request(url=endpoint, method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return response.status is not None
    except urllib.error.HTTPError:
        # Server responded with e.g. 405 -> reachable
        return True
    except Exception:
        return False


# Module-level state to remember fallback resolution for the current run
_ollama_resolved_endpoint = None
_ollama_failed_models = set()
_ollama_resolved_model = None


def resolve_ollama_endpoint(primary_endpoint, verbose=False):
    """Resolve the Ollama endpoint with automatic fallback to localhost.

    Tries the primary endpoint first. If unreachable, falls back to localhost.
    Remembers the resolved endpoint for the current run to avoid re-checking.

    Args:
        primary_endpoint: The primary Ollama endpoint (e.g., '192.168.1.190')
        verbose: Whether to print verbose output

    Returns:
        The resolved endpoint URL to use for this run
    """
    global _ollama_resolved_endpoint
    
    # Return cached result if already resolved
    if _ollama_resolved_endpoint is not None:
        return _ollama_resolved_endpoint
    
    # Normalize primary endpoint
    primary = _normalize_ollama_endpoint(primary_endpoint) if primary_endpoint else None
    fallback = _normalize_ollama_endpoint("localhost")
    
    # Try primary endpoint
    if primary and _ollama_endpoint_reachable(primary):
        _ollama_resolved_endpoint = primary
        if verbose:
            print(f"Ollama endpoint {primary} reachable", file=sys.stderr, flush=True)
        return primary
    
    # Fallback to localhost
    if verbose and primary:
        print(f"Ollama endpoint {primary} unreachable; falling back to {fallback}", file=sys.stderr, flush=True)
    elif verbose:
        print(f"No primary Ollama endpoint configured; using {fallback}", file=sys.stderr, flush=True)
    
    _ollama_resolved_endpoint = fallback
    return fallback


def _fuse_results_with_ollama(results, query, frame_b64s, npz_name, model_id="clef-flash", ollama_endpoint="http://localhost:11434/v1/systemone", threshold=0.5, verbose=False, weight_clip=0.5, weight_ollama=0.5):
    """Use Ollama's decision capability to validate frames individually.

    Every frame in `frame_b64s` is validated against the Ollama decision model.
    `model_id` can be a single model name or a list/comma-separated string;
    models are tried in order per frame until one succeeds.
    The results are then fused using a weighted score of CLIP similarity and Ollama's judgment.
    Only frames whose fused score passes the threshold are kept; failing frames are
    discarded while passing frames are retained with their original metadata.
    The returned list preserves the original result order for passing frames only.
    """
    global _ollama_resolved_model, _ollama_failed_models
    if not frame_b64s:
        print("Warning: No frames were extracted. Skipping Ollama validation.", file=sys.stderr)
        return results

    normalized_endpoint = _normalize_ollama_endpoint(ollama_endpoint)
    model_ids = _as_list(model_id)
    if not model_ids:
        model_ids = ["clef-flash"]

    scored_results = []
    frame_iter = iter(frame_b64s)

    for r in results:
        try:
            frame_b64 = next(frame_iter)
        except StopIteration:
            print("Warning: Frame extraction ended early. Stopping validation.", file=sys.stderr)
            break

        # Use resolved model if available, otherwise try all models
        if _ollama_resolved_model is not None and _ollama_resolved_model in model_ids:
            current_models = [_ollama_resolved_model]
        else:
            current_models = model_ids

        valid_score = None
        last_err = None
        frame_model = None
        models_tried = []
        for model in current_models:
            if model in _ollama_failed_models:
                continue
            models_tried.append(model)
            payload = {
                "model": model,
                "state": f"A user is searching for: {query}",
                "images": [frame_b64],
                "questions": {
                    "is_valid": {
                        "type": "noul",
                        "instructions": f"Does the image match the query: {query}?"
                    }
                }
            }
            
            try:
                req = urllib.request.Request(
                    normalized_endpoint,
                    data=json.dumps(payload).encode("utf-8"),
                    headers={"Content-Type": "application/json"}
                )
                with urllib.request.urlopen(req, timeout=300) as response:
                    res_json = json.loads(response.read().decode("utf-8"))

                valid_score = res_json["answers"]["is_valid"]["noul"]
                if len(models_tried) > 1:
                    # We tried multiple models before finding one that works
                    print(f"Falling back from {models_tried[0]} to '{model}'", file=sys.stderr, flush=True)
                frame_model = model
                break
            except Exception as e:
                last_err = e
                # Only mark model as failed for definitive errors (not timeouts)
                # Timeouts are transient - the server may be busy but still usable
                if "timed out" not in str(e).lower():
                    _ollama_failed_models.add(model)
                if verbose:
                    print(f"Model '{model}' failed for frame at {r['timestamp_seconds']:.1f}s: {e}", file=sys.stderr)

        if valid_score is None:
            if last_err is None:
                last_err = Exception("All Ollama models are unavailable (previously failed for this session)")
            print(f"Error validating frame at {r['timestamp_seconds']:.1f}s: {last_err}. Discarded.", file=sys.stderr)
            continue

        if frame_model and frame_model not in _ollama_failed_models:
            _ollama_resolved_model = frame_model

        # Fused score calculation
        fused_score = (r['score'] * weight_clip) + (valid_score * weight_ollama)
        if fused_score >= threshold:
            print(f"ts={r['timestamp_formatted']} sim={r['score']:.4f} ollama={valid_score:.4f} fused={fused_score:.4f}", file=sys.stderr)
        else:
            print(f"ts={r['timestamp_formatted']} sim={r['score']:.4f} ollama={valid_score:.4f} fused={fused_score:.4f} discard", file=sys.stderr)
        scored_results.append({**r, "fused_score": fused_score, "ollama_score": valid_score})

    # Rerank the results based on fused score
    scored_results.sort(key=lambda x: x["fused_score"], reverse=True)

    # Filter by threshold
    final_results = []
    for res in scored_results:
        if res["fused_score"] >= threshold:
            # Remove the fused_score and ollama_score keys before returning to maintain original structure
            final_results.append({k: v for k, v in res.items() if k not in ["fused_score", "ollama_score"]})
    return final_results




def cmd_search(args):
    cfg = load_config()

    def cfg_val(key, cli_arg):
        return get_config_value(cfg, "search", key, getattr(args, cli_arg, None))

    top_k = cfg_val("top_k", "top_k")
    raw_floor = cfg_val("raw_floor", "raw_floor")
    json_output = cfg_val("json_output", "json")
    save_frames = cfg_val("save_frames", "save_frames")
    verbose = not json_output
    video_path = cfg_val("video", "video")
    fuse = cfg_val("decision_model_fuse", "decision_model_fuse")
    ollama_endpoint = cfg_val("ollama_host", "ollama_host")
    ollama_model = cfg_val("ollama_decision_model", "ollama_model")
    fuse_threshold = cfg_val("fuse_threshold", "fuse_threshold")
    fuse_weight_clip = cfg_val("fuse_weight_clip", "fuse_weight_clip")
    fuse_weight_ollama = cfg_val("fuse_weight_ollama", "fuse_weight_ollama")
    percentile = cfg_val("percentile", "percentile")
    frame_scale = cfg_val("ollama_frame_scale", "ollama_frame_scale")

    # Print config values in verbose mode
    if verbose:
        _models = _as_list(ollama_model)
        pct_str = f", percentile={percentile:.2f}" if percentile and percentile > 0.0 else ""
        print(f"Config: top_k={top_k}, raw_floor={raw_floor:.2f}, json_output={json_output}{pct_str}", file=sys.stderr)
        print(f"        save_frames={save_frames}, video={video_path}, fuse={fuse}, ollama_host={ollama_endpoint}, ollama_model={_models}, fuse_threshold={fuse_threshold}, fuse_weight_clip={fuse_weight_clip:.2f}, fuse_weight_ollama={fuse_weight_ollama:.2f}, ollama_frame_scale={frame_scale:.2f}", file=sys.stderr)

    data = np.load(args.cache, allow_pickle=False)
    embeddings = data["embeddings"]
    timestamps = data["timestamps"]
    cache_model = data.get("model")
    cache_video_path = data.get("video_path")
    stored_md5 = str(data["md5"].item()) if "md5" in data else "N/A"

    if verbose:
        print(f"Cache Info: model={cache_model}, video_path={cache_video_path}, md5={stored_md5}", file=sys.stderr)

    # Update video_path to use the one from cache
    # Note: if user explicitly passed --video, we respect that.
    if args.video is not None:
        video_path = args.video
    else:
        video_path = cache_video_path if cache_video_path else data.get("video_path")

    # MD5 is mandatory in cache
    if "md5" not in data:
        print(
            f"Error: cache '{args.cache}' is missing required 'md5' field. "
            f"Re-index the video to add integrity metadata.",
            file=sys.stderr,
        )
        sys.exit(1)
    stored_md5 = str(data["md5"].item())

    if args.video is not None:
        if not os.path.exists(args.video):
            print(f"Error: --video file '{args.video}' does not exist.", file=sys.stderr)
            sys.exit(1)
        actual_md5 = hashlib.md5()
        with open(args.video, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                actual_md5.update(chunk)
        actual_md5_hex = actual_md5.hexdigest()
        if actual_md5_hex != stored_md5:
            print(
                f"Error: MD5 mismatch for --video '{args.video}'. "
                f"Cache '{args.cache}' was indexed from a different video file "
                f"(stored MD5: {stored_md5}).",
                file=sys.stderr,
            )
            sys.exit(1)
        if verbose:
            print(f"MD5 verified for {args.video}", file=sys.stderr)

    if video_path is not None:
        video_path = str(video_path.item()) if hasattr(video_path, "item") else str(video_path)
        if not os.path.exists(video_path):
            print(f"Error: video file '{video_path}' does not exist in cache '{args.cache}'. "
                      f"Re-index the video or manually provide --video flag when running search.",
                      file=sys.stderr,)
            sys.exit(1)

    # Derive cache name from the .npz filename for result/frame naming
    npz_name = os.path.splitext(os.path.basename(args.cache))[0]

    # Model must match the one used to index the cache. Derive it solely from the
    # cache so a CLI flag can never cause a mismatch (bad/empty results).
    npz_model = None
    if "model" in data:
        try:
            npz_model = str(data["model"].item())
        except Exception:
            pass

    if npz_model is not None:
        model = npz_model
        if verbose:
            print(f"Using model from cache: {model}", file=sys.stderr)
    else:
        print(
            f"Error: cache '{args.cache}' does not record the indexing model "
            f"(missing 'model' field). Re-index the video to add model metadata.",
            file=sys.stderr,
        )
        sys.exit(1)

    engine = VideoSearchEngine(model_id=model)

    # Query: text or image
    # When fuse is enabled, expand candidate pool to top_k * 2 to allow for
    # frames that fail the Ollama fused score check; we keep a reserve buffer
    # capped at top_k extra candidates.
    search_top_k = top_k
    if fuse and top_k is not None and top_k > 0:
        search_top_k = top_k * 2

    if args.text is not None:
        query = args.text
        results = engine.search_text(query, embeddings, timestamps, search_top_k, raw_floor, percentile=percentile)
    elif args.image is not None:
        query = args.image
        results = engine.search_image(query, embeddings, timestamps, search_top_k, raw_floor, percentile=percentile)
    else:
        print("search requires either --text or --image", file=sys.stderr)
        sys.exit(2)

    # Perform Ollama validation first (so failing frames are discarded before saving)
    if fuse:
        # Resolve Ollama endpoint: first reachable from the list (primary first, fallback second)
        ollama_endpoint = resolve_ollama_endpoint(ollama_endpoint, verbose=verbose)
        ollama_models = _as_list(ollama_model)
        if not ollama_models:
            ollama_models = ["clef-flash"]
        if not video_path:
            print("--video is required when using --decision_model_fuse", file=sys.stderr)
            sys.exit(2)

        # Extract frames lazily and validate top-down
        frame_b64_gen = _extract_frames_lazy(results, video_path, verbose, frame_scale=frame_scale)
        results = _fuse_results_with_ollama(results, query, frame_b64_gen, npz_name, model_id=ollama_model, ollama_endpoint=ollama_endpoint, threshold=fuse_threshold, verbose=verbose, weight_clip=fuse_weight_clip, weight_ollama=fuse_weight_ollama)

    # If fuse was enabled, results were already sorted/filtered by fused score;
    # slice back to the requested top_k so we don't return more than the user asked for.
    if fuse and top_k is not None and top_k > 0:
        results = results[:top_k]


    # Save frames for remaining (validated) results only
    if save_frames:
        if not video_path:
            print("--video is required when using --save-frames", file=sys.stderr)
            sys.exit(2)

        os.makedirs(save_frames, exist_ok=True)
        saved_count, frame_paths = _save_matched_frames(results, npz_name, video_path, save_frames, verbose)
        if verbose:
            print(f"Saved {saved_count} frame(s) to {save_frames}", file=sys.stderr)

    # Output to stdout (JSON or text)
    if json_output:
        # Ensure all metadata values are JSON serializable (avoid numpy types)
        def _safe(v):
            if v is None:
                return None
            if isinstance(v, np.ndarray):
                return str(v)
            if isinstance(v, (np.integer,)):
                return int(v)
            if isinstance(v, (np.floating,)):
                return float(v)
            if isinstance(v, (bytes,)):
                return v.decode("utf-8", errors="replace")
            return v

        payload = {
            "query": query,
            "results": results,
            "metadata": {
                "top_k": _safe(top_k),
                "raw_floor": _safe(raw_floor),
                "json_output": _safe(json_output),
                "save_frames": _safe(save_frames),
                "video": _safe(video_path),
                "fuse": _safe(fuse), 
                "ollama_host": _safe(ollama_endpoint),
                "ollama_model": _safe(ollama_model),
                "fuse_threshold": _safe(fuse_threshold),
                "fuse_weight_clip": _safe(fuse_weight_clip),
                "fuse_weight_ollama": _safe(fuse_weight_ollama),
                "ollama_frame_scale": _safe(frame_scale),
                "percentile": _safe(percentile),
                "cache_model": _safe(cache_model),
                "cache_video_path": _safe(cache_video_path),
                "cache_md5": _safe(stored_md5),
            },
        }
        print(json.dumps(payload))
    else:
        if results:
            print(_format_results_text(results))
        else:
            print("No matches found.")


def main():
    parser = argparse.ArgumentParser(prog="vsearch", description="Semantic video timestamp search (offline CLIP).")
    sub = parser.add_subparsers(dest="command", required=True)

    p_index = sub.add_parser("index", help="Extract keyframes and build a .npz cache")
    p_index.add_argument("video", nargs="+", help="Input video file(s), directory(ies), or glob pattern(s) (e.g. '*.mp4', './videos/', 'vid1.mp4 ./dir/*.mp4')")
    p_index.add_argument("--output", "-o", default=None, help="Path to save the output cache archive (.npz). If a directory, auto-generates filename: {video_name}_{model_name}_{sample_rate}s_{md5_prefix8}.npz")
    p_index.add_argument("--sample-rate", "-r", type=int, default=None, help="Frame sampling interval in seconds (default: 1)")
    p_index.add_argument("--batch-size", "-bs", type=int, default=None, help="Batch size for frame encoding during indexing (default: 32)")
    p_index.add_argument("--model", "-m",default=None, help="CLIP model variant (default: openai/clip-vit-base-patch32)")
    p_index.add_argument("--recursive", "-R", action="store_true", default=None, help="Recursively scan directories for video files")
    p_index.add_argument("--ext", "-e", default=None, help="Comma-separated video file extensions to match in directories (default: mp4,avi,mov,mkv,webm,flv,wmv)")
    p_index.add_argument("--dry-run", action="store_true", default=None, help="List the video files that would be indexed without processing them")
    
    p_index.set_defaults(func=cmd_index)

    p_search = sub.add_parser("search", help="Query a pre-computed cache")
    p_search.add_argument("cache", help="Path to pre-computed embeddings cache (.npz)")
    p_search.add_argument("--text", "-t", default=None, help="Natural language search prompt")
    p_search.add_argument("--image", "-i", default=None, help="Path to reference image file")
    p_search.add_argument("--top-k", "-k", type=int, default=None, help="Maximum number of top results to return (default: all)")
    p_search.add_argument("--json", action="store_true", default=None, help="Output results in strict JSON format")
    p_search.add_argument("--raw-floor", "-rf", type=float, default=None, help="Absolute noise floor for raw similarity (default: 0.25)")
    p_search.add_argument("--percentile", "-p", type=float, default=None, help="Percentile cutoff (0.0-1.0, e.g., 0.2 keeps top 20%% of frames). Setting this enables percentile filtering automatically.")
    
    p_search.add_argument("--save-frames", "-sf", default=None, help="Directory to save matched frame images as {cache_name}_{HHMMSS}_{score}.jpg")
    p_search.add_argument("--video", "-vid", default=None, help="Path to original video file (required with --save-frames)")
    p_search.add_argument("--decision-model-fuse","-dmf", action="store_true", default=None, help="Use local Ollama decision model to fuse top results. Results failing the fused score threshold are discarded from match results.")
    p_search.add_argument("--fuse-threshold","-ft", type=float, default=None, help="Threshold for the fused score (default: 0.5). Results with score < threshold are discarded.")
    p_search.add_argument("--fuse-weight-clip","-fwc", type=float, default=None, help="Weight given to CLIP similarity in fused score (default: 0.3).")
    p_search.add_argument("--fuse-weight-ollama","-fwo", type=float, default=None, help="Weight given to Ollama judgment in fused score (default: 0.7).")
    p_search.add_argument("--ollama-host","-oh", default=None, help="Ollama server addr (default: 192.168.1.190). Falls back to localhost if unreachable")
    p_search.add_argument("--ollama-frame-scale","-ofs", type=float, default=None, help="Scale factor for frames sent to Ollama decision model (0.0-1.0, default: 0.2). Lower values = faster inference")
    p_search.add_argument("--ollama-model","-dm", default=None, help="Ollama decision model name(s), comma-separated fallback list tried in order for validation")
    p_search.set_defaults(func=cmd_search)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
