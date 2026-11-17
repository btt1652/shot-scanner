"""Command-line interface for vsearch (decision-model-only mode).

This CLI searches videos using an Ollama decision model as the sole judge.
No CLIP embeddings or .npz caches are required.
"""
import argparse
import base64
import hashlib
import io
import json
import os
import subprocess
import sys
import threading
import time
import urllib.request
import cv2
import numpy as np
from PIL import Image

from .config import load_config, get_config_value

DEFAULT_VIDEO_EXTENSIONS = [".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv"]


def _hash_video_content(video_path, chunk_mb=8):
    """Hash first and last N MB of video file using BLAKE2b. Returns raw digest bytes."""
    size = os.path.getsize(video_path)
    chunk = chunk_mb * 1024 * 1024
    h = hashlib.blake2b(digest_size=32)
    with open(video_path, "rb") as f:
        h.update(f.read(min(chunk, size)))
        if size > 2 * chunk:
            f.seek(-chunk, os.SEEK_END)
            h.update(f.read(chunk))
    return h.digest()


def _hash_params(sample_rate, frame_scale, state, instructions, model):
    """Hash search params (sample-rate, frame-scale, state, instructions, model)."""
    s = f"{sample_rate}|{frame_scale}|{state}|{instructions}|{model}"
    return hashlib.blake2b(s.encode("utf-8"), digest_size=32).digest()


def _cache_filename(video_path, sample_rate, frame_scale, state, instructions, model):
    """Build cache filename: <single_hash>.json from content + params."""
    content_bytes = _hash_video_content(video_path)
    params_bytes = _hash_params(sample_rate, frame_scale, state, instructions, model)
    h = hashlib.blake2b()
    h.update(content_bytes)
    h.update(params_bytes)
    return f"{h.hexdigest()[:16]}.json"


def _try_load_cache(cache_dir, video_path, sample_rate, frame_scale, state, instructions, model, decision_model_floor):
    """Try to load cached results.

    Returns (results, data) where:
      - Complete cache (total_extracted >= total_sampled): results = floor-filtered all_results, data = full cache dict.
      - Incomplete cache (cancelled run): results = None, data = full cache dict (caller resumes from frame_index).
      - Cache miss/unreadable: results = None, data = None.

    Filters all_results by decision_model_floor at load time so cache is
    reusable across different floor thresholds.
    """
    if not cache_dir:
        return None, None
    fname = _cache_filename(video_path, sample_rate, frame_scale, state, instructions, model)
    cache_file = os.path.join(cache_dir, fname)
    print(f"Cache lookup: {cache_file}", file=sys.stderr)
    if not os.path.exists(cache_file):
        print(f"Cache miss: {cache_file}", file=sys.stderr)
        return None, None
    try:
        with open(cache_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        all_results = data.get("all_results")
        if all_results is None:
            all_results = data.get("results", [])
        # total_sampled is the number of frames actually sampled (not raw video frames)
        total_sampled = data.get("total_sampled")
        total_extracted = data.get("total_extracted")
        if total_sampled is not None and total_extracted is not None:
            if total_extracted < total_sampled:
                print(f"Cache incomplete: {cache_file} (extracted={total_extracted}/{total_sampled}) — resuming", file=sys.stderr)
                return None, data
        results = [r for r in all_results if r.get("score", 0) >= decision_model_floor]
        print(f"Cache hit: {cache_file} ({len(results)} results after floor filter, {len(all_results)} total)", file=sys.stderr)
        return results, data
    except (json.JSONDecodeError, OSError) as e:
        print(f"Warning: Failed to read cache file {cache_file}: {e}", file=sys.stderr)
        return None, None


def _save_cache(cache_dir, video_path, sample_rate, frame_scale, state, instructions, model, all_results, metadata, total_sampled=None, total_validated=None, total_extracted=None, frame_index=None, passing_count=None):
    """Save all results (including resume state) to a cache JSON file."""
    if not cache_dir:
        return
    fname = _cache_filename(video_path, sample_rate, frame_scale, state, instructions, model)
    cache_file = os.path.join(cache_dir, fname)
    try:
        os.makedirs(cache_dir, exist_ok=True)
        data = {
            "video_path": video_path,
            "query": state or instructions or "",
            "parameters": {
                "decision_model_sample_rate": sample_rate,
                "ollama_frame_scale": frame_scale,
                "ollama_state_template": state,
                "ollama_instructions_template": instructions,
                "ollama_model": model,
            },
            "all_results": all_results,
            "metadata": metadata,
            "total_sampled": total_sampled,
            "total_validated": total_validated,
            "total_extracted": total_extracted,
            "frame_index": frame_index,
            "passing_count": passing_count,
        }
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        print(f"cache saved {cache_file})", file=sys.stderr)
    except OSError as e:
        print(f"Warning: Failed to save cache file {cache_file}: {e}", file=sys.stderr)


def _resolve_video_paths(inputs, recursive=False, extensions=None):
    """Resolve a list of inputs (files, directories, glob patterns) into concrete video file paths."""
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

        if os.path.isdir(inp):
            walker = os.walk(inp) if recursive else [(inp, [], os.listdir(inp))]
            for root, dirs, files in walker:
                for fname in files:
                    if fname.lower().endswith(exts_lower):
                        _add(os.path.join(root, fname))
                        matched_any = True

        elif any(c in inp for c in "*?[]"):
            for m in sorted(glob.glob(inp, recursive=recursive)):
                if os.path.isfile(m):
                    _add(m)
                    matched_any = True

        else:
            if os.path.isfile(inp):
                _add(inp)
                matched_any = True

        if not matched_any:
            missing.append(inp)

    return resolved, missing


def _normalize_ollama_endpoint(endpoint):
    """Normalize an Ollama endpoint to a full URL."""
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


def _hash_video_content(video_path, chunk_mb=8):
    """Hash first and last N MB of video file using BLAKE2b. Returns raw digest bytes."""
    size = os.path.getsize(video_path)
    chunk = chunk_mb * 1024 * 1024
    h = hashlib.blake2b(digest_size=32)
    with open(video_path, "rb") as f:
        h.update(f.read(min(chunk, size)))
        if size > 2 * chunk:
            f.seek(-chunk, os.SEEK_END)
            h.update(f.read(chunk))
    return h.digest()

def _ollama_endpoint_reachable(endpoint, timeout=5.0):
    """Check if the Ollama endpoint is reachable."""
    endpoint = _normalize_ollama_endpoint(endpoint)
    if not endpoint:
        return False
    try:
        req = urllib.request.Request(url=endpoint, method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return response.status is not None
    except urllib.error.HTTPError:
        return True
    except Exception:
        return False

_ollama_resolved_endpoint = None

def resolve_ollama_endpoint(primary_endpoint):
    """Resolve the Ollama endpoint with automatic fallback to localhost."""
    global _ollama_resolved_endpoint
    if _ollama_resolved_endpoint is not None:
        return _ollama_resolved_endpoint

    primary = _normalize_ollama_endpoint(primary_endpoint) if primary_endpoint else None
    fallback = _normalize_ollama_endpoint("localhost")

    if primary and _ollama_endpoint_reachable(primary):
        _ollama_resolved_endpoint = primary
        print(f"Ollama endpoint {primary} reachable", file=sys.stderr, flush=True)
        return primary

    if primary:
        print(f"Ollama endpoint {primary} unreachable; falling back to {fallback}", file=sys.stderr, flush=True)
    else:
        print(f"No primary Ollama endpoint configured; using {fallback}", file=sys.stderr, flush=True)

    _ollama_resolved_endpoint = fallback
    return fallback

def _extract_frame_at(video_path, timestamp, out_path):
    """Extract a single frame at given timestamp and save to out_path as JPEG."""
    cmd = [
        'ffmpeg', '-y', '-v', 'error',
        '-ss', str(timestamp),
        '-i', video_path,
        '-frames:v', '1',
        out_path
    ]
    try:
        subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        return True
    except subprocess.CalledProcessError:
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
        cv2.imwrite(out_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return True
    except Exception:
        return False

def _save_matched_frames(results, video_path, output_dir):
    """Extract and save matched frames as JPEG images."""
    os.makedirs(output_dir, exist_ok=True)
    saved = 0
    frame_paths = []
    for r in results:
        ts = r.get("start_seconds", r.get("timestamp_seconds"))
        score = r.get("score_max", r.get("score"))
        if ts >= 3600:
            ts_formatted = f"{int(ts)//3600:02d}:{int(ts)%3600//60:02d}:{int(ts)%60:02d}"
        else:
            ts_formatted = f"{int(ts)//60:02d}:{int(ts)%60:02d}"
        vp = r.get("source_video_path") or video_path
        if vp:
            video_name = os.path.splitext(os.path.basename(vp))[0]
        else:
            video_name = "video"
        out_path = os.path.join(output_dir, f"{video_name}_{ts_formatted}_{score:.4f}.jpg")
        if _extract_frame_at(vp, ts, out_path):
            saved += 1
            frame_paths.append(out_path)
            print(f"Saved frame at {ts:.1f}s -> {out_path}", file=sys.stderr)
        else:
            print(f"Failed to extract frame at {ts:.1f}s from {vp}", file=sys.stderr)
    return saved, frame_paths


def cmd_search(args):
    cfg = load_config()

    def cfg_val(key, cli_arg):
        return get_config_value(cfg, "search", key, getattr(args, cli_arg, None))

    top_k = cfg_val("top_k", "top_k")
    global_k = cfg_val("global_k", "global_k")
    json_output = cfg_val("json_output", "json")
    save_frames = cfg_val("save_frames", "save_frames")
    decision_model_floor = cfg_val("decision_model_floor", "decision_model_floor")
    decision_model_sample_rate = cfg_val("decision_model_sample_rate", "decision_model_sample_rate")
    ollama_endpoint = cfg_val("ollama_host", "ollama_host")
    ollama_model = cfg_val("ollama_decision_model", "ollama_model")
    frame_scale = cfg_val("ollama_frame_scale", "ollama_frame_scale")
    recursive = cfg_val("recursive", "recursive")
    state_template = cfg_val("ollama_state_template", "ollama_state")
    instructions_template = cfg_val("ollama_instructions_template", "ollama_instructions")
    cache_dir = cfg_val("cache_dir", "cache")
    save_cache_dir = cfg_val("save_cache_dir", "save_cache")
    still_frame_threshold = cfg_val("still_frame_threshold", "still_frame_threshold") or 0.95
    same_scene_threshold = cfg_val("same_scene_threshold", "same_scene_threshold") or 0.80
    max_scene_batch_size = cfg_val("max_scene_batch_size", "max_scene_batch_size") or 4
    max_scene_batch_size = min(max_scene_batch_size, 4)

    # Apply defaults if config/CLI left templates empty
    if not state_template:
        state_template = ""
    if not instructions_template:
        instructions_template = ""

    # Apply defaults if config/CLI left templates empty
    if not state_template:
        state_template = ""
    if not instructions_template:
        instructions_template = ""

    # State/instructions are the query — use state for cache identity
    query = state_template or instructions_template or ""

    # Resolve video input from positional args
    video_input = getattr(args, "video_paths", None)
    if not video_input:
        print("Error: video path(s) required", file=sys.stderr)
        sys.exit(1)

    video_paths, missing = _resolve_video_paths(video_input, recursive=recursive)
    if not video_paths:
        print(f"Error: no video files found from input '{video_input}'", file=sys.stderr)
        sys.exit(1)
    for m in missing:
        print(f"Warning: no video matched input '{m}'", file=sys.stderr)

    # Resolve Ollama endpoint once
    ollama_endpoint = resolve_ollama_endpoint(ollama_endpoint)

    # Ollama timeout (seconds) — configurable via config or CLI
    ollama_timeout = cfg_val("ollama_timeout", "ollama_timeout")

    print(f"Config: top_k={top_k}, global_k={global_k}, json_output={json_output}", file=sys.stderr)
    print(f"        save_frames={save_frames}, video={video_paths[0]}, decision_model_floor={decision_model_floor}, decision_model_sample_rate={decision_model_sample_rate}, ollama_host={ollama_endpoint}, ollama_model={ollama_model}, ollama_frame_scale={frame_scale:.2f}, ollama_state={state_template!r}, ollama_instructions={instructions_template!r}, ollama_timeout={ollama_timeout}", file=sys.stderr)

    sample_rate = decision_model_sample_rate if decision_model_sample_rate else 1

    model = ollama_model if ollama_model else "clef-flash"  # Single model only (no fallback)

    batch = []
    all_results = []
    final_results = []
    passing_count = 0
    total_extracted = 0
    total_validated = 0
    instantiate_save = bool(save_frames) and not (
        global_k is not None and global_k > 0 and
        top_k is not None and top_k > 0
    )

    def _scaled_bgr(frame_bgr):
        """Return BGR numpy resized to frame_scale (same as _b64_from_bgr scales)."""
        scale = frame_scale if frame_scale else 0.2
        h, w = frame_bgr.shape[:2]
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        return cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)

    def _b64_from_bgr(frame_bgr):
        """Encode a BGR numpy frame to scaled JPEG base64."""
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        scale = frame_scale if frame_scale else 0.2
        new_w = max(1, int(pil_img.width * scale))
        new_h = max(1, int(pil_img.height * scale))
        pil_img = pil_img.resize((new_w, new_h), Image.LANCZOS)
        buffered = io.BytesIO()
        pil_img.save(buffered, format="JPEG", quality=80)
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    def _validate_batch_single(frames_batch):
        """Send all frames in a batch in one multi-image Ollama request; returns results with shared batch score."""
        results = []
        passing = 0
        if not frames_batch:
            return results, passing

        ts_list = ", ".join(f"{int(ts)//3600:02d}:{int(ts)%3600//60:02d}:{int(ts)%60:02d}" for ts, _, _ in frames_batch)
        print(f"send request [{ts_list}]", file=sys.stderr)

        images = [b64 for _, b64, _ in frames_batch]
        payload = {
            "model": model,
            "state": state_template or "",
            "images": images,
            "questions": {
                "is_valid": {
                    "type": "noul",
                    "instructions": instructions_template or ""
                }
            }
        }
        try:
            req = urllib.request.Request(
                ollama_endpoint,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=ollama_timeout) as response:
                res_json = json.loads(response.read().decode("utf-8"))
            batch_score = res_json["answers"]["is_valid"]["noul"]
            if isinstance(batch_score, list):
                batch_score = max(batch_score) if batch_score else 0.0
        except urllib.error.URLError as e:
            print(f"Batch request failed: {e}. Discarding batch.", file=sys.stderr)
            return results, passing
        except Exception as e:
            print(f"Batch request error: {e}. Discarding batch.", file=sys.stderr)
            return results, passing

        for ts, b64, vp in frames_batch:
            ts_formatted = f"{int(ts)//3600:02d}:{int(ts)%3600//60:02d}:{int(ts)%60:02d}"
            results.append({
                "timestamp_seconds": ts,
                "timestamp_formatted": ts_formatted,
                "score": batch_score,
                "source_video_path": vp,
            })
            if batch_score >= decision_model_floor:
                passing += 1
                if instantiate_save:
                    _save_matched_frames([{
                        "timestamp_seconds": ts,
                        "timestamp_formatted": ts_formatted,
                        "score": batch_score,
                        "source_video_path": vp,
                    }], vp, save_frames)

        pass_fail = "pass" if batch_score >= decision_model_floor else "fail"
        print(f"response [{ts_list}] score={batch_score:.2f} {pass_fail}", file=sys.stderr)

        return results, passing

    def _flush_batch():
        nonlocal batch, total_validated, passing_count, video_prev_score
        if not batch:
            return
        out, passing = _validate_batch_single(batch)
        all_results.extend(out)
        passing_count += passing
        total_validated += len(batch)
        batch = []
        video_prev_score = out[0]["score"] if out else video_prev_score
        # Persist validated results immediately so a cancelled run can resume
        if save_cache_dir:
            video_results = [r for r in all_results if r.get("source_video_path") == vp]
            if video_results:
                # Sort results by timestamp to ensure correct ordering in cache
                video_results.sort(key=lambda x: x["timestamp_seconds"])
                _save_cache(
                    save_cache_dir, vp,
                    sample_rate, frame_scale,
                    state_template, instructions_template, model_for_cache,
                    video_results, {},
                    total_sampled=total_sampled,
                    total_validated=total_validated,
                    total_extracted=total_extracted,
                    frame_index=frame_index,
                    passing_count=passing_count,
                )

    # Cache lookup when --cache flag provided (CLI overrides config)
    cache_dir = getattr(args, "cache", None) or cache_dir
    # Save-cache directory (--save-cache). Defaults to None (no save) if flag not used.
    save_cache_dir = getattr(args, "save_cache", None) or save_cache_dir

    model_for_cache = model

    # Process each video: check cache first, then search if miss
    for vp in video_paths:
        frame_index = 0  # default if no state

        video_prev_score = 0.0  # score of most recently validated frame for this video

        # Check cache (sample mode only). Cache stores both results and resume state.
        if cache_dir:
            cache_results, cache_data = _try_load_cache(
                cache_dir, vp,
                sample_rate, frame_scale,
                state_template, instructions_template, model_for_cache,
                decision_model_floor,
            )
            if cache_results is not None:
                all_results.extend(cache_results)
                passing_count += len(cache_results)
                print(f"Loaded {len(cache_results)} result(s) from cache for {vp}", file=sys.stderr)
                continue
            # Cache incomplete - resume from stored frame index if available
            if cache_data is not None:
                frame_index = int(cache_data.get("frame_index", 0))
                passing_count = int(cache_data.get("passing_count", 0))
                total_extracted = int(cache_data.get("total_extracted", 0))
                total_validated = int(cache_data.get("total_validated", 0))
                print(f"Resuming from cache: frame_index={frame_index}, validated={total_validated}", file=sys.stderr)

        cap = cv2.VideoCapture(vp)
        if not cap.isOpened():
            print(f"Warning: Cannot open video {vp}, skipping", file=sys.stderr)
            continue

        print(f"Processing video: {vp}", file=sys.stderr)

        prev_small_gray = None
        prev_ts = None

        # Get total frame count (if available)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.get(cv2.CAP_PROP_FRAME_COUNT) is not None else None

        # --- Blind uniform sampling mode ---
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps is None or fps <= 0:
            fps = 30.0
        stride = max(1, int(fps * sample_rate))
        total_sampled = int((total_frames + stride - 1) / stride) if total_frames is not None else 0

        # Seek to resume position if cache provided one
        if frame_index > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break

            if frame_index % stride == 0:
                small_bgr = _scaled_bgr(frame_bgr)
                small_gray = cv2.cvtColor(small_bgr, cv2.COLOR_BGR2GRAY)

                b64 = _b64_from_bgr(frame_bgr)
                ts = frame_index / fps

                skip_frame = True

                if prev_small_gray is not None:
                    diff = cv2.absdiff(prev_small_gray, small_gray)
                    mean_diff = float(np.mean(diff))
                    similarity = 1.0 - (mean_diff / 255.0)

                    ts_prev_formatted = f"{int(prev_ts)//3600:02d}:{int(prev_ts)%3600//60:02d}:{int(prev_ts)%60:02d}"
                    ts_cur_formatted = f"{int(ts)//3600:02d}:{int(ts)%3600//60:02d}:{int(ts)%60:02d}"

                    if similarity >= still_frame_threshold:
                        skip_frame = True
                        print(f"sim check {ts_cur_formatted} <-> {ts_prev_formatted} sim={similarity:.2f} (still frame, skipped)", file=sys.stderr)
                        # Crash-safe: immediately append skipped frame with inherited score so it is persisted to cache
                        all_results.append({
                            "timestamp_seconds": ts,
                            "timestamp_formatted": ts_cur_formatted,
                            "score": video_prev_score,
                            "source_video_path": vp,
                        })
                        total_extracted += 1
                        if video_prev_score >= decision_model_floor:
                            passing_count += 1
                    elif similarity >= same_scene_threshold and len(batch) < max_scene_batch_size:
                        print(f"sim check {ts_cur_formatted} <-> {ts_prev_formatted} sim={similarity:.2f} (same scene)", file=sys.stderr)
                        batch.append((ts, b64, vp))
                        total_extracted += 1
                        batch_ts = ", ".join(f"{int(t)//3600:02d}:{int(t)%3600//60:02d}:{int(t)%60:02d}" for t, _, _ in batch)
                        print(f"add {ts_cur_formatted} to request [{batch_ts}]", file=sys.stderr)
                        if len(batch) >= max_scene_batch_size:
                            _flush_batch()
                        skip_frame = False
                    else:
                        print(f"sim check {ts_cur_formatted} <-> {ts_prev_formatted} sim={similarity:.2f} (new scene)", file=sys.stderr)
                        _flush_batch()
                        batch = [(ts, b64, vp)]
                        total_extracted += 1
                        skip_frame = False
                else:
                    batch = [(ts, b64, vp)]
                    total_extracted += 1
                    skip_frame = False
                    ts_formatted = f"{int(ts)//3600:02d}:{int(ts)%3600//60:02d}:{int(ts)%60:02d}"
                    print(f"add {ts_formatted} to request [{ts_formatted}]", file=sys.stderr)

                if not skip_frame:
                    prev_small_gray = small_gray
                    prev_ts = ts

                if total_extracted % 10 == 0:
                    # Show progress as extracted/total_sampled so at end it reads extracted=720/720
                    print(f"{vp} (extracted={total_extracted}/{total_sampled}, validated={total_validated}, passing={passing_count})", file=sys.stderr)

            frame_index += 1

        cap.release()

        # Flush remaining batch for this video
        _flush_batch()

        # Save cache for this video (includes resume state)
        if save_cache_dir:
            video_results = [r for r in all_results if r.get("source_video_path") == vp]
            if video_results:
                # Sort results by timestamp to ensure correct ordering in cache
                video_results.sort(key=lambda x: x["timestamp_seconds"])
                _save_cache(
                    save_cache_dir, vp,
                    sample_rate, frame_scale,
                    state_template, instructions_template, model_for_cache,
                    video_results, {},
                    total_sampled=total_sampled,
                    total_validated=total_validated,
                    total_extracted=total_extracted,
                    frame_index=frame_index,
                    passing_count=passing_count,
                )

    # Filter by floor, then select top-k, then order chronologically
    final_results = [r for r in all_results if r["score"] >= decision_model_floor]
    final_results.sort(key=lambda x: x["score"], reverse=True)
    if global_k is not None and global_k > 0:
        final_results = final_results[:global_k]
    elif top_k is not None and top_k > 0:
        final_results = final_results[:top_k]
    final_results.sort(key=lambda x: (x.get("source_video_path", ""), x["timestamp_seconds"]))

    # Save matched frames
    if save_frames and final_results:
        saved_count, _paths = _save_matched_frames(final_results, video_paths[0], save_frames)
        print(f"Saved {saved_count} frame(s) to {save_frames}", file=sys.stderr)

    # Output
    if json_output:
        def _safe(v):
            if isinstance(v, (np.integer,)):
                return int(v)
            if isinstance(v, (np.floating,)):
                return float(v)
            return v

        serializable = [{k: _safe(v) for k, v in r.items()} for r in final_results]
        payload = {
            "query": query,
            "results": serializable,
            "metadata": {
                "decision_model_floor": _safe(decision_model_floor),
                "decision_model_sample_rate": _safe(sample_rate),
                "ollama_host": _safe(ollama_endpoint),
                "ollama_model": _safe(model),
                "ollama_frame_scale": _safe(frame_scale),
                "top_k": _safe(top_k),
                "global_k": _safe(global_k),
                "video": _safe(video_paths[0]),
            },
        }
        print(json.dumps(payload))
    else:
        if final_results:
            for r in final_results:
                if 'timestamp_formatted' in r:
                    # Raw result
                    print(f"ts={r['timestamp_formatted']}  score={r['score']:.4f}")
                else:
                    # Merged range
                    print(f"ts={r['start_formatted']}-{r['end_formatted']}  score={r['score_max']:.4f}  (max={r["score_max"]:.4f}, avg={r["score_avg"]:.4f}, n={r["sample_count"]})")
        else:
            print("No matches found.")


def main():
    parser = argparse.ArgumentParser(
        prog="vsearch",
        description="Semantic video timestamp search using an Ollama decision model.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_search = sub.add_parser("search", help="Search videos using the decision model")
    p_search.add_argument("video_paths", nargs="+", help="One or more video files, directories, or glob patterns (e.g., video1.mp4 video2/*.mp4)")

    p_search.add_argument("--top-k", "-k", type=int, default=None, help="Maximum number of top results to return (default: all passing)")
    p_search.add_argument("--global-k", "-gk", type=int, default=None, help="Global top-K across all videos, overriding --top-k")
    p_search.add_argument("--json", action="store_true", default=None, help="Output results in strict JSON format")
    p_search.add_argument("--save-frames", "-sf", default=None, help="Directory to save matched frame images as {video_name}_{HHMMSS}_{score}.jpg")
    p_search.add_argument("--decision-model-floor", "-dmt", type=float, default=None, help="Threshold for Ollama score (default: 0.5). Results below are discarded.")
    p_search.add_argument("--decision-model-sample-rate", "-dmsr", type=int, default=None, help="Frame sampling interval in seconds (default: 1)")
    p_search.add_argument("--ollama-host", "-oh", default=None, help="Ollama server address (default: 192.168.1.190). Falls back to localhost if unreachable")
    p_search.add_argument("--ollama-frame-scale", "-ofs", type=float, default=None, help="Scale factor for frames sent to Ollama (0.0-1.0, default: 0.2). Lower = faster")
    p_search.add_argument("--ollama-model", "-dm", default=None, help="Ollama model name(s), comma-separated fallback list tried in order")
    p_search.add_argument("--ollama-timeout", "-ot", type=int, default=None, help="Ollama request timeout in seconds (default: 60)")
    p_search.add_argument("--still-frame-threshold", type=float, default=None, help="Similarity (0-1) above which a frame is considered still (default: 0.95)")
    p_search.add_argument("--same-scene-threshold", type=float, default=None, help="Similarity (0-1) above which frames are considered same scene (default: 0.80)")
    p_search.add_argument("--max-scene-batch-size", type=int, default=None, help="Maximum number of consecutive same-scene frames per Ollama request (default: 4, capped by Ollama limit)")
    p_search.add_argument("--recursive", "-R", action="store_true", default=False, help="Recursively scan directories for video files")
    p_search.add_argument("--cache", "-c", type=str, default=None, help="Path to cache directory to look up existing cache files. Overrides config 'cache_dir'. Only used when this flag is given")
    p_search.add_argument("--save-cache", "-sc", type=str, default=None, nargs="?", const=".", help="Path to cache directory for saving search results. Overrides config 'save_cache_dir'. Defaults to current directory (.) if used without a path")
    p_search.add_argument("--ollama-state", "-s", type=str, default=None, help="State context string for Ollama (default: empty)")
    p_search.add_argument("--ollama-instructions", "-oi", type=str, default=None, help="Instructions string for Ollama (default: empty)")
    p_search.set_defaults(func=cmd_search)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
