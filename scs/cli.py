"""Command-line interface for scs (decision-model-only mode).

This CLI classifies video frames using an Ollama decision model as the sole judge.
No CLIP embeddings or .npz caches are required.

Questions are defined as a JSON array (via CLI or config.toml) and each frame gets
one answer row per question stored in the SQLite cache.
"""
import argparse
import base64
import glob
import io
import json
import os
import signal
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path
import cv2
import numpy as np
from PIL import Image

from .config import load_config, get_config_value, get_questions
from .cache import CacheDB, _content_hash, _params_hash

DEFAULT_VIDEO_EXTENSIONS = [".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv"]

# Ollama supports at most 4 images per multi-image request, so scene batches are fixed at 4.
MAX_SCENE_BATCH_SIZE = 4

_COLORS = {
    "reset": "\033[0m",
    "red": "\033[31m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "blue": "\033[34m",
    "cyan": "\033[36m",
    "gray": "\033[90m",
    "bold": "\033[1m",
}

def _cache_params(sample_rate, frame_scale, state, questions_json, model, same_scene_threshold=None, still_frame_threshold=None):
    """Build the parameter dict used for the cache key."""
    params = {
        "sample_rate": sample_rate,
        "frame_scale": frame_scale,
        "state": state,
        "questions_json": questions_json,
        "model": model,
    }
    if same_scene_threshold is not None:
        params["same_scene_threshold"] = same_scene_threshold
    if still_frame_threshold is not None:
        params["still_frame_threshold"] = still_frame_threshold
    return params


def _try_load_cache(cache, video_path, sample_rate, frame_scale, state, questions_json, model, same_scene_threshold=None, still_frame_threshold=None):
    """Try to load cached results for a video + params (via analysis run).

    Returns (answers, data) where:
      - Cache hit (complete/partial): answers = frame answer rows, data includes run_id, cache_status, max_frame_index, max_batch_id
      - Cache miss/incomplete: answers = None, data = None
    """
    if not cache:
        return None, None
    try:
        cache_db = CacheDB(cache)
        params_hash = _params_hash(**_cache_params(
            sample_rate, frame_scale, state, questions_json, model,
            same_scene_threshold, still_frame_threshold,
        ))
        content_hash = _content_hash(video_path)

        video = cache_db.get_video(video_path)
        if video is None:
            print(f"Cache miss: video {video_path} not in cache", file=sys.stderr)
            return None, None

        run = cache_db.get_run(video["id"], params_hash)
        if run is None:
            print(f"Cache miss: no run for video {video_path} with content={content_hash} params={params_hash}", file=sys.stderr)
            return None, None

        answers = cache_db.get_answers(run["id"])
        if not answers:
            print(f"Cache miss: no answers for {video_path}", file=sys.stderr)
            return None, None

        # Determine cache status
        cache_status = run.get("cache_status", "complete")

        # Get resume metadata from stored answers
        max_frame = max(a["frame_index"] for a in answers)
        max_batch = max(a.get("batch_id", 0) for a in answers)

        resume_data = {
            "run_id": run["id"],
            "video_id": video["id"],
            "video_path": video_path,
            "cache_status": cache_status,
            "max_frame_index": max_frame,
            "max_batch_id": max_batch,
        }

        print(f"Cache hit: {video_path} (run={run['id']}, {len(answers)} answer rows, status={cache_status})", file=sys.stderr)
        return answers, resume_data
    except Exception as e:
        print(f"Warning: Failed to read cache: {e}", file=sys.stderr)
        return None, None


def _save_cache(cache, video_path, sample_rate, frame_scale, state, questions_json, model, all_results, total_sampled=None, same_scene_threshold=None, still_frame_threshold=None, status="complete"):
    """Save all answer rows to SQLite cache.

    Args:
        status: 'complete', 'partial', or 'incomplete'. Controls whether the
            next run treats this video as fully done or resumes from it.
    """
    if not cache:
        return
    try:
        cache_db = CacheDB(cache)
        params_hash = _params_hash(**_cache_params(
            sample_rate, frame_scale, state, questions_json, model,
            same_scene_threshold, still_frame_threshold,
        ))
        content_hash = _content_hash(video_path)

        # Upsert video metadata
        video_id = cache_db.upsert_video(
            video_path=video_path,
            content_hash=content_hash,
        )

        # Upsert analysis run for this param set
        run_id = cache_db.upsert_run(
            video_id=video_id,
            params_hash=params_hash,
            model=model,
            questions_json=questions_json,
        )

        cache_db.upsert_frame_answers(run_id, all_results)

        # Mark run status (complete vs partial for resumable checkpoints)
        cache_db.update_run_status(run_id, status)

        print(f"cache saved {video_path} ({len(all_results)} answer rows, status={status})", file=sys.stderr)
    except Exception as e:
        print(f"Warning: Failed to save cache: {e}", file=sys.stderr)
def _parse_criteria(criteria_str):
    """Parse criteria from a string. Accepts JSON dict or multiline 'key: description' format."""
    if not criteria_str:
        return {}
    criteria_str = criteria_str.strip()
    if not criteria_str:
        return {}
    try:
        obj = json.loads(criteria_str)
        if isinstance(obj, dict):
            return obj
    except (json.JSONDecodeError, ValueError):
        pass
    result = {}
    for line in criteria_str.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" in line:
            key, _, desc = line.partition(":")
            key = key.strip()
            desc = desc.strip()
            if key:
                result[key] = desc
    return result


def _normalize_questions(raw_questions):
    """Validate and normalize question definitions.

    Each raw question must have 'name' (string) and 'criteria' (dict or string).
    Returns a list of dicts with keys: name, instructions, criteria (dict).
    """
    if not isinstance(raw_questions, list):
        raise ValueError("questions must be a JSON array")
    normalized = []
    for idx, q in enumerate(raw_questions):
        name = q.get("name")
        instructions = q.get("instructions")
        criteria = q.get("criteria")
        if not name:
            raise ValueError(f"question {idx} missing 'name'")
        if criteria is None:
            raise ValueError(f"question {name} missing 'criteria'")
        if isinstance(criteria, str):
            criteria = _parse_criteria(criteria)
        if not isinstance(criteria, dict) or not criteria:
            raise ValueError(f"question {name} criteria must be a dict")
        # Check for duplicate question names
        existing_names = [n['name'] for n in normalized]
        if name in existing_names:
            raise ValueError(f"duplicate question name '{name}'. Question names must be unique.")
        normalized.append({
            "name": name,
            "instructions": instructions or "",
            "criteria": criteria,
        })
    return normalized


def _check_and_confirm_overwrite(cache, video_paths, questions, params_hash):
    """Check if video already has cached answers for any of the questions.
    If so, warn user and ask for confirmation before proceeding.
    Only flags as duplicate if question name AND params_hash match (same model+config).
    """
    cache_db = CacheDB(cache)
    question_names = [q["name"] for q in questions]
    overlapping = []  # (vp, qname, count, status, run_id)

    for vp in video_paths:
        video = cache_db.get_video(vp)
        if video is None:
            continue
        # Get runs for this video
        runs = cache_db.list_runs(vp)
        for run in runs:
            # Only consider runs with same params_hash (same model+config)
            if run.get("params_hash") != params_hash:
                continue
            status = run.get("cache_status", "empty")
            # If status is 'partial', we should RESUME, not prompt for overwrite.
            # Only prompt if status is 'complete' (or 'empty' but has answers).
            if status == "partial":
                continue
            # Check if any answer matches our question names
            for qname in question_names:
                cur = cache_db.conn.execute(
                    "SELECT COUNT(*) FROM frame_answers WHERE run_id = ? AND question_name = ?",
                    (run["id"], qname)
                )
                count = cur.fetchone()[0]
                if count > 0:
                    overlapping.append((vp, qname, count, status, run["id"]))

    cache_db.close()

    if not overlapping:
        return

    print("Warning: Found existing cached answers that overlap with new analysis:", file=sys.stderr)
    for vp, qname, count, status, _ in overlapping[:10]:
        print(f"  {vp}: question='{qname}' has {count} answers (status={status})", file=sys.stderr)

    # Ask user for confirmation
    while True:
        try:
            ans = input("Overwrite existing results? (y/n): ").strip().lower()
        except EOFError:
            ans = "n"
        if ans in ("y", "yes"):
            # Delete overlapping answers
            cache_db = CacheDB(cache)
            for vp, qname, count, status, run_id in overlapping:
                cache_db.conn.execute(
                    "DELETE FROM frame_answers WHERE run_id = ? AND question_name = ?",
                    (run_id, qname)
                )
            cache_db.conn.commit()
            cache_db.close()
            print("Existing overlapping answers deleted. Proceeding with new analysis.", file=sys.stderr)
            return
        elif ans in ("n", "no"):
            print("Aborted by user.", file=sys.stderr)
            sys.exit(0)
        else:
            print("Please answer 'y' or 'n'.", file=sys.stderr)


def _questions_json(questions):
    """Serialize normalized questions to a canonical JSON string (sorted)."""
    return json.dumps(questions, sort_keys=True, ensure_ascii=False)


def _sql_with_params(sql: str, params: list) -> str:
    """Replace ? placeholders with actual values for display purposes."""
    # Collapse whitespace: newlines/tabs -> single space
    compact = ' '.join(sql.split())
    result = compact
    for param in params:
        if param is None:
            result = result.replace("?", "NULL", 1)
        elif isinstance(param, str):
            escaped = param.replace("'", "''")
            result = result.replace("?", f"'{escaped}'", 1)
        elif isinstance(param, bool):
            result = result.replace("?", "TRUE" if param else "FALSE", 1)
        else:
            result = result.replace("?", str(param), 1)
    return result


def _validate_batch_classify(frames_batch, model, state_template, questions, ollama_timeout, ollama_endpoint, batch_id_counter=None):
    """Validate a batch of frames using multiple choice questions.

    frames_batch: list of tuples (frame_index, timestamp_seconds, b64_image, video_path)
    Returns (answer_rows, next_batch_id).
    """
    if not frames_batch:
        return [], 0
    batch_id = (batch_id_counter or 0) + 1
    ts_list = ", ".join(f"{int(ts)//3600:02d}:{int(ts)%3600//60:02d}:{int(ts)%60:02d}" for _, ts, _, _ in frames_batch)
    print(f"{_COLORS.get("yellow")}send request [{ts_list}]{_COLORS.get("reset")}", file=sys.stderr)

    images = [b64 for _, _, b64, _ in frames_batch]
    questions_payload = {}
    for q in questions:
        questions_payload[q["name"]] = {
            "type": "choice",
            "instructions": q["instructions"],
            "criteria": q["criteria"],
        }
    payload = {
        "model": model,
        "state": state_template or "",
        "images": images,
        "questions": questions_payload,
    }
    try:
        req = urllib.request.Request(ollama_endpoint, data=json.dumps(payload).encode("utf-8"), headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=ollama_timeout) as response:
            res_json = json.loads(response.read().decode("utf-8"))
        answers = res_json.get("answers", {})
    except urllib.error.URLError as e:
        print(f"Classify request failed: {e}. Discarding batch.", file=sys.stderr)
        return [], batch_id
    except Exception as e:
        print(f"Classify request error: {e}. Discarding batch.", file=sys.stderr)
        return [], batch_id

    answer_rows = []
    for frame_index, ts, _b64, vp in frames_batch:
        ts_formatted = f"{int(ts)//3600:02d}:{int(ts)%3600//60:02d}:{int(ts)%60:02d}"
        for q in questions:
            qname = q["name"]
            q_ans = answers.get(qname, {})
            probs = q_ans.get("probabilities", {}) if isinstance(q_ans, dict) else {}
            if not isinstance(probs, dict):
                probs = {}
            top_choice = None
            top_prob = 0.0
            for ch, p in probs.items():
                try:
                    pv = float(p)
                except (TypeError, ValueError):
                    pv = 0.0
                if top_choice is None or pv > top_prob:
                    top_prob = pv
                    top_choice = ch
            # Extract confidence from Ollama response (if provided)
            # Expected format: {"choice": "value", "probability": 0.9, "confidence": 0.85}
            confidence = None
            if isinstance(q_ans, dict):
                # Try different possible keys for confidence
                for conf_key in ["confidence", "certainty", "score", "model_confidence"]:
                    if conf_key in q_ans:
                        try:
                            confidence = float(q_ans[conf_key])
                            break
                        except (TypeError, ValueError):
                            pass
            
            answer_rows.append({
                "frame_index": frame_index,
                "timestamp_seconds": ts,
                "timestamp_formatted": ts_formatted,
                "source_video_path": vp,
                "batch_id": batch_id,
                "question_name": qname,
                "question_instructions": q["instructions"],
                "criteria_json": json.dumps(q["criteria"], sort_keys=True, ensure_ascii=False),
                "top_choice": top_choice,
                "top_probability": top_prob,
                "confidence": confidence,
                "probabilities_json": json.dumps(probs, sort_keys=True, ensure_ascii=False),
                "raw_response": json.dumps(q_ans, ensure_ascii=False) if isinstance(q_ans, dict) else None,
            })
    print(f"{_COLORS.get("yellow")}response [{ts_list}] batch_id={batch_id} questions={[q['name'] for q in questions]}{_COLORS.get("reset")}", file=sys.stderr)
    return answer_rows, batch_id
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


def cmd_analyze(args):
    """Classify video frames using an Ollama choice-question model."""
    cfg = load_config()

    def cfg_val(key, cli_arg):
        return get_config_value(cfg, key, getattr(args, cli_arg, None))

    # Read config values (removed: top_k, global_k, json_output, decision_model_floor, merge_range, top_choice)
    decision_model_sample_rate = cfg_val("decision_model_sample_rate", "decision_model_sample_rate")
    ollama_endpoint = cfg_val("ollama_host", "ollama_host")
    ollama_model = cfg_val("ollama_decision_model", "ollama_model")
    frame_scale = cfg_val("ollama_frame_scale", "ollama_frame_scale")
    recursive = cfg_val("recursive", "recursive")
    state_template = cfg_val("ollama_state_template", "ollama_state")
    cache = cfg_val("cache", "cache")
    still_frame_threshold = cfg_val("still_frame_threshold", "still_frame_threshold") or 0.95
    same_scene_threshold = cfg_val("same_scene_threshold", "same_scene_threshold") or 0.80
    ollama_timeout = cfg_val("ollama_timeout", "ollama_timeout")

    if not state_template:
        state_template = ""

    # Get questions from config or CLI --questions
    cli_questions = getattr(args, "questions", None)
    raw_questions = get_questions(cfg, cli_questions)
    questions = _normalize_questions(raw_questions)
    questions_json = _questions_json(questions)

    if not questions:
        print("Error: no questions defined — provide questions in config.toml or via --questions", file=sys.stderr)
        sys.exit(1)

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

    ollama_endpoint = resolve_ollama_endpoint(ollama_endpoint)

    print(f"Config: sample_rate={decision_model_sample_rate}, frame_scale={frame_scale:.2f}", file=sys.stderr)
    print(f"        model={ollama_model}, host={ollama_endpoint}, state={state_template!r}, timeout={ollama_timeout}", file=sys.stderr)
    print(f"        questions={[q['name'] for q in questions]}", file=sys.stderr)

    sample_rate = decision_model_sample_rate if decision_model_sample_rate else 1
    model = ollama_model if ollama_model else "typesafe/jev-1.14"

    # Build params_hash for duplicate detection (includes model, thresholds, etc.)
    params_hash = _params_hash(**_cache_params(
        sample_rate, frame_scale, state_template, questions_json, model,
        same_scene_threshold, still_frame_threshold,
    ))

    # Check for existing cached answers for these videos/questions
    if cache:
        _check_and_confirm_overwrite(cache, video_paths, questions, params_hash)

    batch = []
    all_results = []
    total_extracted = 0
    total_validated = 0
    batch_id_counter = 0

    def _scaled_bgr(frame_bgr):
        scale = frame_scale if frame_scale else 0.2
        h, w = frame_bgr.shape[:2]
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        return cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)

    def _b64_from_bgr(frame_bgr):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        scale = frame_scale if frame_scale else 0.2
        new_w = max(1, int(pil_img.width * scale))
        new_h = max(1, int(pil_img.height * scale))
        pil_img = pil_img.resize((new_w, new_h), Image.LANCZOS)
        buffered = io.BytesIO()
        pil_img.save(buffered, format="JPEG", quality=80)
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    def _flush_batch():
        nonlocal batch, total_validated, batch_id_counter
        if not batch:
            return
        out, batch_id_counter = _validate_batch_classify(
            batch, model, state_template, questions, ollama_timeout, ollama_endpoint, batch_id_counter
        )
        all_results.extend(out)
        total_validated += len(batch)
        batch = []

    save_cache = cache

    # Register signal handler for graceful Ctrl+C / SIGINT mid-run
    _shutdown_saving = False
    def _signal_handler(signum, frame):
        nonlocal _shutdown_saving
        if _shutdown_saving:
            return  # Prevent re-entry
        _shutdown_saving = True
        print("\nInterrupted — saving partial progress before exit...", file=sys.stderr)
        if save_cache and all_results:
            # Save all accumulated results across all videos processed so far
            # Note: we save per-video, so loop through unique video paths seen
            seen_videos = set(r.get("source_video_path") for r in all_results)
            for seen_vp in seen_videos:
                vid_results = [r for r in all_results if r.get("source_video_path") == seen_vp]
                if vid_results:
                    vid_results.sort(key=lambda x: x["timestamp_seconds"])
                    _save_cache(
                        save_cache, seen_vp,
                        sample_rate, frame_scale,
                        state_template, questions_json, model,
                        vid_results,
                        total_sampled=total_sampled,
                        same_scene_threshold=same_scene_threshold,
                        still_frame_threshold=still_frame_threshold,
                        status="partial",
                    )
        print("Progress saved. Exiting.", file=sys.stderr)
        sys.exit(130)

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    # Process each video: check cache first, then classify if miss
    for vp in video_paths:
        # Per-video counters (reset for each video)
        total_extracted = 0
        total_validated = 0
        batch_id_counter = 0
        total_sent = 0
        
        frame_index = 0
        if cache:
            cache_results, cache_data = _try_load_cache(
                cache, vp,
                sample_rate, frame_scale,
                state_template, questions_json, model,
                same_scene_threshold, still_frame_threshold,
            )
            if cache_results is not None:
                all_results.extend(cache_results)
                print(f"Loaded {len(cache_results)} answer row(s) from cache for {vp}", file=sys.stderr)
                if cache_data is not None:
                    cache_status = cache_data.get("cache_status", "complete")
                    if cache_status == "partial":
                        # Resume from where we left off
                        frame_index = cache_data.get("max_frame_index", 0)
                        batch_id_counter = cache_data.get("max_batch_id", 0)
                        print(f"Resuming from frame_index={frame_index}, validated={batch_id_counter} batches", file=sys.stderr)
                    elif cache_status == "complete":
                        # Full cache hit - skip entire video processing
                        print(f"Full cache hit for {vp}, skipping processing", file=sys.stderr)
                        frame_index = -1  # sentinel to skip processing below
                    else:
                        print(f"Cache incomplete for {vp} - resuming from frame 0", file=sys.stderr)
                else:
                    print(f"Cache incomplete for {vp} - no data available, resuming from frame 0", file=sys.stderr)
            else:
                print(f"Cache incomplete for {vp} - no answers found, resuming from frame 0", file=sys.stderr)
        else:
            frame_index = 0

        if frame_index < 0:
            # Already fully cached or nothing to do, skip to next video
            continue

        cap = cv2.VideoCapture(vp)
        if not cap.isOpened():
            print(f"Warning: Cannot open video {vp}, skipping", file=sys.stderr)
            continue

        print(f"Processing video: {vp}", file=sys.stderr)
        prev_small_gray = None
        prev_ts = None

        if frame_index > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            print(f"Resuming from frame_index={frame_index}", file=sys.stderr)

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.get(cv2.CAP_PROP_FRAME_COUNT) is not None else None
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps is None or fps <= 0:
            fps = 30.0
        stride = max(1, int(fps * sample_rate))
        total_sampled = int((total_frames + stride - 1) / stride) if total_frames is not None else 0

        if frame_index > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

        total_skipped = 0  # still frames skipped

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
                        total_extracted += 1
                        total_skipped += 1
                    elif similarity >= same_scene_threshold and len(batch) < MAX_SCENE_BATCH_SIZE:
                        print(f"sim check {ts_cur_formatted} <-> {ts_prev_formatted} sim={similarity:.2f} (same scene)", file=sys.stderr)
                        batch.append((frame_index, ts, b64, vp))
                        total_extracted += 1
                        batch_ts = ", ".join(f"{int(t)//3600:02d}:{int(t)%3600//60:02d}:{int(t)%60:02d}" for _, t, _, _ in batch)
                        print(f"add {ts_cur_formatted} to request [{batch_ts}]", file=sys.stderr)
                        if len(batch) >= MAX_SCENE_BATCH_SIZE:
                            _flush_batch()
                        skip_frame = False
                    else:
                        print(f"sim check {ts_cur_formatted} <-> {ts_prev_formatted} sim={similarity:.2f} (new scene)", file=sys.stderr)
                        _flush_batch()
                        batch = [(frame_index, ts, b64, vp)]
                        total_extracted += 1
                        skip_frame = False
                else:
                    batch = [(frame_index, ts, b64, vp)]
                    total_extracted += 1
                    skip_frame = False
                    ts_formatted = f"{int(ts)//3600:02d}:{int(ts)%3600//60:02d}:{int(ts)%60:02d}"
                    print(f"add {ts_formatted} to request [{ts_formatted}]", file=sys.stderr)

                if not skip_frame:
                    prev_small_gray = small_gray
                    prev_ts = ts

                if total_extracted % 10 == 0:
                    print(f"{_COLORS.get("green")}{vp} (sampled={total_extracted}/{total_sampled}, skipped={total_skipped}, validated={total_validated}){_COLORS.get("reset")}", file=sys.stderr)

                # Periodic checkpoint save every 100 extracted frames
                if total_extracted % 100 == 0 and save_cache:
                    video_results = [r for r in all_results if r.get("source_video_path") == vp]
                    if video_results:
                        video_results.sort(key=lambda x: x["timestamp_seconds"])
                        _save_cache(
                            save_cache, vp,
                            sample_rate, frame_scale,
                            state_template, questions_json, model,
                            video_results,
                            total_sampled=total_sampled,
                            same_scene_threshold=same_scene_threshold,
                            still_frame_threshold=still_frame_threshold,
                            status="partial",
                        )

            frame_index += 1

        cap.release()

        # Flush remaining batch for this video
        _flush_batch()

        # Save cache for this video
        if save_cache:
            video_results = [r for r in all_results if r.get("source_video_path") == vp]
            if video_results:
                video_results.sort(key=lambda x: x["timestamp_seconds"])
                _save_cache(
                    save_cache, vp,
                    sample_rate, frame_scale,
                    state_template, questions_json, model,
                    video_results,
                    total_sampled=total_sampled,
                    same_scene_threshold=same_scene_threshold,
                    still_frame_threshold=still_frame_threshold,
                )

    # Output summary
    print(f"Processed {len(video_paths)} video(s): {len(all_results)} answer rows saved to cache.", file=sys.stderr)
    if all_results:
        # Print a quick summary by question
        from collections import defaultdict
        summary = defaultdict(lambda: defaultdict(int))
        for r in all_results:
            q = r.get("question_name")
            tc = r.get("top_choice")
            summary[q][tc] += 1
        for q, counts in summary.items():
            print(f"  {q}: {dict(counts)}", file=sys.stderr)
        print(f"Use: scs cache sql \"SELECT * FROM frame_answers WHERE question_name='{questions[0]['name']}'\" --cache {cache or './.cache.db'}", file=sys.stderr)
    else:
        print("No answers generated.", file=sys.stderr)


def _print_table(rows):
    """Print a list of dicts as an aligned text table."""
    if not rows:
        print("(no rows)")
        return
    cols = list(rows[0].keys())
    widths = {c: max(len(str(c)), max(len(str(r.get(c, ""))) for r in rows)) for c in cols}
    print("  ".join(str(c).ljust(widths[c]) for c in cols))
    print("  ".join("-" * widths[c] for c in cols))
    for r in rows:
        print("  ".join(str(r.get(c, "")).ljust(widths[c]) for c in cols))


def cmd_query(args):
    """Run a SQL query on the cache, optionally extract matching frames."""
    import cv2
    import os

    cache_db = CacheDB(args.cache or "./.cache.db")

    sql = args.sql
    if not sql:
        print("Error: SQL query required", file=sys.stderr)
        sys.exit(1)

    try:
        rows = cache_db.execute_sql(sql)
    except Exception as e:
        print(f"Error executing SQL: {e}", file=sys.stderr)
        cache_db.close()
        sys.exit(1)

    save_frame_dir = args.save_frame
    if save_frame_dir:
        # Group by video to batch seeks
        from collections import defaultdict
        by_video = defaultdict(list)

        # Build mapping from run_id to video_path via analysis_runs
        run_to_path = {}
        for r in cache_db.execute_sql("SELECT ar.id AS run_id, v.video_path FROM analysis_runs ar JOIN videos v ON ar.video_id = v.id"):
            run_to_path[r["run_id"]] = r["video_path"]

        # Add source_video_path to each row
        for row in rows:
            run_id = row.get("run_id")
            if run_id and run_id in run_to_path:
                row["source_video_path"] = run_to_path[run_id]
                by_video[run_to_path[run_id]].append(row)

        if not by_video:
            print("No matching frames to extract.", file=sys.stderr)
            cache_db.close()
            sys.exit(0)

        os.makedirs(save_frame_dir, exist_ok=True)

        saved = 0
        fmt = "jpg"
        quality = 90

        for video_path, frames in by_video.items():
            if not os.path.exists(video_path):
                print(f"Warning: video not found: {video_path}, skipping", file=sys.stderr)
                continue

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"Warning: cannot open video: {video_path}", file=sys.stderr)
                continue

            frames_sorted = sorted(frames, key=lambda r: r.get("frame_index", 0))

            for frame_info in frames_sorted:
                fi = frame_info.get("frame_index", 0)
                cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
                ret, frame = cap.read()
                if not ret:
                    print(f"Warning: could not read frame {fi} from {video_path}", file=sys.stderr)
                    continue

                video_name = os.path.splitext(os.path.basename(video_path))[0]
                q_name = frame_info.get("question_name", "frame")
                choice = frame_info.get("top_choice", "unknown")
                ts = frame_info.get("timestamp_seconds", 0)
                fname = f"{video_name}_f{fi:06d}_{ts:.1f}s_{q_name}_{choice}.{fmt}"
                out_path = os.path.join(save_frame_dir, fname)

                if fmt == "jpg":
                    cv2.imwrite(out_path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
                elif fmt == "png":
                    cv2.imwrite(out_path, frame)
                else:
                    cv2.imwrite(out_path, frame)
                saved += 1
                print(f"Saved frame: {out_path}", file=sys.stderr)

            cap.release()

        print(f"Extracted {saved} frame(s) to {save_frame_dir}", file=sys.stderr)
        cache_db.close()
        sys.exit(0)

    _print_table(rows)
    cache_db.close()
    sys.exit(0)

def cmd_search(args):
    """Search cached frame answers by question name, choice, and probability.

    Usage:
        scs search <video> [--q "question=choice:prob"]* [--top N] [--top-per-video N] [--merge-result-threshold SEC] [--model MODEL --save-frame DIR] [--stitch [PATH]]

    Each --q adds a filter: 'question_name=choice:min_probability'. All must match for a row to qualify.
    Example:
        scs search video.mp4 \
          --q "activity=yes:0.8" \
          --q "time_of_day=no:0.9" \
          --top 10 \
          --save-frame ./frames
    """
    import cv2
    import os

    cache_db = CacheDB(args.cache or "./.cache.db")

    # Parse --q flags into list of (question_name, choice, min_probability)
    q_list = [q.strip() for q in (args.q_values or []) if q.strip()]

    # Check for duplicate --q arguments
    if len(q_list) != len(set(q_list)):
        seen = {}
        for q in q_list:
            if q in seen:
                print(f"Error: duplicate --q argument '{q}'. Each --q must be unique.", file=sys.stderr)
                cache_db.close()
                sys.exit(1)
            seen[q] = True

    # Parse conditions with field-OR support
    # Syntax per condition: question=choice:prob[:confidence]  (AND between prob and confidence)
    # OR between prob and confidence: question=choice:prob|conf  (e.g., activity=yes:0.8|0.7)
    # Multiple --q flags are AND'd together
    # Example: --q "activity=yes:0.8|0.7" --q "scene_type=couple:0.9"
    #   => (activity=yes AND (prob>=0.8 OR confidence>=0.7)) AND scene_type=couple
    conditions = []  # list of (qname, choice, prob, confidence, use_or)
    for q in q_list:
        parts = q.split('=')
        if len(parts) < 2:
            print(f"Error: invalid --q format '{q}'. Expected 'question=choice:prob[:confidence]' or 'question=choice:prob|conf'.", file=sys.stderr)
            cache_db.close()
            sys.exit(1)
        qname = parts[0].strip()
        rest = parts[1]  # e.g., "yes:0.8|0.7" or "yes:0.8:0.7"
        
        # Check for field-OR syntax (prob|conf)
        if '|' in rest:
            # Format: choice:prob|conf
            choice_part, rest_part = rest.split(':', 1)
            choice = choice_part.strip().lower()
            prob_conf = rest_part.split('|')
            if len(prob_conf) != 2:
                print(f"Error: invalid --q format '{q}'. Expected 'choice:prob|conf' for field-OR.", file=sys.stderr)
                cache_db.close()
                sys.exit(1)
            try:
                prob = float(prob_conf[0].strip())
                confidence = float(prob_conf[1].strip())
            except ValueError:
                print(f"Error: invalid probability/confidence in '{q}'. Use numeric values.", file=sys.stderr)
                cache_db.close()
                sys.exit(1)
            conditions.append((qname, choice, prob, confidence, True))  # use_or = True
        else:
            # Standard format: choice:prob[:confidence] (AND between prob and confidence)
            rest_parts = rest.split(':')
            if len(rest_parts) < 2 or len(rest_parts) > 3:
                print(f"Error: invalid --q format '{q}'. Expected 'choice:prob' or 'choice:prob:confidence' or 'choice:prob|conf'.", file=sys.stderr)
                cache_db.close()
                sys.exit(1)
            choice = rest_parts[0].strip().lower()
            try:
                prob = float(rest_parts[1].strip())
                confidence = float(rest_parts[2].strip()) if len(rest_parts) == 3 else None
            except ValueError:
                print(f"Error: invalid probability/confidence in '{q}'. Use numeric values.", file=sys.stderr)
                cache_db.close()
                sys.exit(1)
            conditions.append((qname, choice, prob, confidence, False))  # use_or = False

    top_n = int(args.top) if args.top is not None else None

    # Build SQL - we need self-join for multiple question conditions on same frame
    # Use frame_answers joined against itself for each condition
    if not conditions:
        print("Error: at least one --q filter required", file=sys.stderr)
        cache_db.close()
        sys.exit(1)

    # Build query with self-joins for multiple conditions
    # First condition uses fa (main frame_answers), subsequent use fa1, fa2, etc.
    select_cols = "fa.run_id, fa.frame_index, fa.timestamp_seconds, fa.question_name, fa.top_choice, fa.top_probability, fa.confidence, v.id AS video_id"
    from_clause = "frame_answers fa JOIN analysis_runs ar1 ON fa.run_id = ar1.id"
    where_clauses = []
    params = []

    # Build SQL with field-OR support (prob OR confidence per condition)
    # Each condition can be AND (prob AND confidence) or OR (prob OR confidence)
    # Multiple conditions are AND'd together
    
    for (qname, choice, prob, confidence, use_or) in conditions:
        if use_or:
            # OR between prob and confidence: (prob >= ? OR confidence >= ?)
            clause = "fa.question_name = ? AND fa.top_choice = ? AND (fa.top_probability >= ? OR fa.confidence >= ?)"
            params.extend([qname, choice, prob, confidence])
        else:
            # AND between prob and confidence (default)
            clause = "fa.question_name = ? AND fa.top_choice = ? AND fa.top_probability >= ?"
            params.extend([qname, choice, prob])
            if confidence is not None:
                clause += " AND fa.confidence >= ?"
                params.append(confidence)
        where_clauses.append(clause)
    
    # Simple join: frame_answers -> analysis_runs -> videos
    from_clause = "frame_answers fa JOIN analysis_runs ar1 ON fa.run_id = ar1.id LEFT JOIN videos v ON ar1.video_id = v.id"

    # Limit to specified video(s) - support hash-based lookup
    # For each input, try exact path match first, then fall back to content_hash match
    # If hash matches but path differs, update the video_path in DB to the new path
    video_paths = args.video if isinstance(args.video, list) else [args.video]
    matched_video_ids = []
    for vp in video_paths:
        # Try exact path match first
        cur = cache_db.conn.execute(
            "SELECT id, content_hash FROM videos WHERE video_path = ?", (vp,)
        )
        row = cur.fetchone()
        if row:
            matched_video_ids.append(row["id"])
            print(f"Matched by path: '{vp}' -> video_id={row['id']}", file=sys.stderr)
            continue
        # Fallback: compute hash of input and match by content_hash
        try:
            input_hash = _content_hash(vp)
            cur = cache_db.conn.execute(
                "SELECT id, video_path FROM videos WHERE content_hash = ?", (input_hash,)
            )
            row = cur.fetchone()
            if row:
                # Hash matches but path differs - update to new path
                if row["video_path"] != vp:
                    print(f"Hash match: updating video path '{row['video_path']}' -> '{vp}'", file=sys.stderr)
                    cache_db.conn.execute(
                        "UPDATE videos SET video_path = ?, updated_at = datetime('now') WHERE id = ?",
                        (vp, row["id"])
                    )
                    cache_db.conn.commit()
                matched_video_ids.append(row["id"])
                print(f"Matched by hash: '{vp}' -> video_id={row['id']} (cached path: '{row['video_path']}')", file=sys.stderr)
                continue
        except (FileNotFoundError, IOError):
            pass
        # No match found
        print(f"Warning: no cached video matches '{vp}'", file=sys.stderr)

    print("", file=sys.stderr)  # blank line between cache check and SQL

    if matched_video_ids:
        video_id_placeholders = ','.join(['?'] * len(matched_video_ids))
        where_clauses.append(f"v.id IN ({video_id_placeholders})")
        params.extend(matched_video_ids)
    else:
        # No valid videos - add impossible condition to return empty
        where_clauses.append("1=0")

    # Filter by model name if specified
    if getattr(args, "model", None):
        cur = cache_db.conn.execute(
            "SELECT DISTINCT id FROM analysis_runs WHERE model = ?", (args.model,)
        )
        model_run_ids = [r[0] for r in cur.fetchall()]
        if not model_run_ids:
            print(f"Error: no analysis runs found for model '{args.model}'", file=sys.stderr)
            cache_db.close()
            sys.exit(1)
        where_clauses.append(f"fa.run_id IN ({','.join(['?'] * len(model_run_ids))})")
        params.extend(model_run_ids)

    # Filter by specific run_id if specified
    if getattr(args, "run_id", None) is not None:
        where_clauses.append("fa.run_id = ?")
        params.append(args.run_id)

    top_per_video = getattr(args, "top_per_video", None)
    merge_threshold = getattr(args, "merge_result_threshold", None)

    # --top-per-video overrides --top when both are supplied
    if top_per_video is not None and top_n is not None:
        top_n = None

    use_merge = merge_threshold is not None

    def _merge_frame_rows_into_ranges(rows, threshold):
        """Merge consecutive rows with same score into ranges."""
        from collections import defaultdict
        groups = defaultdict(list)
        for r in rows:
            key = (r.get("run_id"), r.get("question_name"))
            groups[key].append(r)

        ranges = []
        for (run_id, q_name), grp in groups.items():
            grp_sorted = sorted(
                grp,
                key=lambda r: (r.get("timestamp_seconds") or 0, r.get("frame_index") or 0),
            )
            current = None
            for r in grp_sorted:
                prob = r.get("top_probability")
                ts = r.get("timestamp_seconds") or 0
                fid = r.get("frame_index") or 0
                if (
                    current is not None
                    and prob == current.get("top_probability")
                    and (ts - current.get("end_timestamp")) <= threshold
                ):
                    current["end_timestamp"] = ts
                    current["frame_indices"].append(fid)
                else:
                    if current is not None:
                        ranges.append(current)
                    current = {
                        "run_id": run_id,
                        "question_name": q_name,
                        "top_choice": r.get("top_choice"),
                        "top_probability": prob,
                        "confidence": r.get("confidence"),
                        "start_timestamp": ts,
                        "end_timestamp": ts,
                        "frame_indices": [fid],
                        "video_id": r.get("video_id"),
                    }
            if current is not None:
                ranges.append(current)

        ranges.sort(key=lambda x: (x.get("top_probability") or 0), reverse=True)
        return ranges

    if use_merge:
        # Fetch all matching rows; limits are applied to merged ranges in Python.
        query_sql = f"""
            SELECT DISTINCT {select_cols}
            FROM {from_clause}
            WHERE {' AND '.join(where_clauses)}
            ORDER BY fa.top_probability DESC
        """
        print(f"{_sql_with_params(query_sql, params)}\n", file=sys.stderr)
        rows = cache_db.execute_sql(query_sql, params)

        results = _merge_frame_rows_into_ranges(rows, merge_threshold)

        if top_n is not None:
            results = sorted(results, key=lambda r: (r.get("top_probability") or 0), reverse=True)[:top_n]

        if top_per_video is not None:
            from collections import defaultdict
            by_run = defaultdict(list)
            for r in results:
                by_run[r.get("run_id")].append(r)
            results = []
            for rid, group in by_run.items():
                group_sorted = sorted(group, key=lambda r: (r.get("top_probability") or 0), reverse=True)
                results.extend(group_sorted[:top_per_video])

        rows = results
    else:
        if top_per_video is not None:
            # Use a window function to limit the top N frames per video.
            query_sql = f"""
                WITH ranked AS (
                    SELECT
                        fa.run_id,
                        fa.frame_index,
                        fa.timestamp_seconds,
                        fa.question_name,
                        fa.top_choice,
                        fa.top_probability,
                        v.id AS video_id,
                        ROW_NUMBER() OVER (PARTITION BY v.id ORDER BY fa.top_probability DESC) AS rn
                    FROM {from_clause}
                    WHERE {' AND '.join(where_clauses)}
                )
                SELECT run_id, frame_index, timestamp_seconds, question_name, top_choice, top_probability
                FROM ranked
                WHERE rn <= ?
                ORDER BY top_probability DESC
            """
            params.append(top_per_video)
            print(f"{_sql_with_params(query_sql, params)}\n", file=sys.stderr)
            rows = cache_db.execute_sql(query_sql, params)
        else:
            limit_expr = f"LIMIT {top_n}" if top_n is not None else ""

            query_sql = f"""
                SELECT DISTINCT {select_cols}
                FROM {from_clause}
                WHERE {' AND '.join(where_clauses)}
                ORDER BY fa.top_probability DESC
                {limit_expr}
            """
            print(f"{_sql_with_params(query_sql, params)}\n", file=sys.stderr)
            rows = cache_db.execute_sql(query_sql, params)

    # Sort final search results by timestamp ascending by default.
    def _result_timestamp(r):
        ts = r.get("start_timestamp") if r.get("start_timestamp") is not None else r.get("timestamp_seconds")
        return ts if ts is not None else 0

    rows = sorted(rows, key=lambda r: ( _result_timestamp(r), r.get("run_id") or 0 ))

    save_frame_dir = args.save_frame
    if save_frame_dir:
        # Group by video to batch seeks
        from collections import defaultdict
        by_video = defaultdict(list)

        # Build mapping from run_id to video_path via analysis_runs
        run_to_path = {}
        for r in cache_db.execute_sql("SELECT ar.id AS run_id, v.video_path FROM analysis_runs ar JOIN videos v ON ar.video_id = v.id"):
            run_to_path[r["run_id"]] = r["video_path"]

        # Add source_video_path to each row
        for row in rows:
            run_id = row.get("run_id")
            if run_id and run_id in run_to_path:
                row["source_video_path"] = run_to_path[run_id]
                by_video[run_to_path[run_id]].append(row)

        if not by_video:
            print("No matching frames found.", file=sys.stderr)
            cache_db.close()
            sys.exit(0)

        os.makedirs(save_frame_dir, exist_ok=True)

        saved = 0
        fmt = "jpg"
        quality = 90

        for video_path, frames in by_video.items():
            if not os.path.exists(video_path):
                print(f"Warning: video not found: {video_path}, skipping", file=sys.stderr)
                continue

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"Warning: cannot open video: {video_path}", file=sys.stderr)
                continue

            def _frame_key(r):
                return r.get("frame_index") or (r.get("frame_indices") or [0])[0]

            frames_sorted = sorted(frames, key=_frame_key)

            for frame_info in frames_sorted:
                fi = frame_info.get("frame_index") or (frame_info.get("frame_indices") or [None])[0]
                if fi is None:
                    continue
                cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
                ret, frame = cap.read()
                if not ret:
                    print(f"Warning: could not read frame {fi} from {video_path}", file=sys.stderr)
                    continue

                video_name = os.path.splitext(os.path.basename(video_path))[0]
                q_name = frame_info.get("question_name", "frame")
                choice = frame_info.get("top_choice", "unknown")
                ts = frame_info.get("timestamp_seconds") or frame_info.get("start_timestamp") or 0
                fname = f"{video_name}_f{fi:06d}_{ts:.1f}s_{q_name}_{choice}.jpg"
                out_path = os.path.join(save_frame_dir, fname)

                cv2.imwrite(out_path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
                saved += 1
                print(f"Saved frame: {out_path}", file=sys.stderr)

            cap.release()

        print(f"Extracted {saved} frame(s) to {save_frame_dir}", file=sys.stderr)
        cache_db.close()
        sys.exit(0)

    if getattr(args, "stitch", False):
        # Stitch feature: use ffmpeg to merge matched frames into a single video
        import subprocess
        import os
        import shutil
        import tempfile
        from collections import defaultdict

        cfg = load_config()
        # --stitch now uses --merge-result-threshold (default 1.0 seconds)
        threshold = getattr(args, "merge_result_threshold", None)
        if threshold is None:
            threshold = 1.0

        # Build mapping run_id -> video_path and add source_video_path to each row
        run_to_path = {}
        for r in cache_db.execute_sql("SELECT ar.id AS run_id, v.video_path FROM analysis_runs ar JOIN videos v ON ar.video_id = v.id"):
            run_to_path[r["run_id"]] = r["video_path"]
        for row in rows:
            run_id = row.get("run_id")
            if run_id and run_id in run_to_path and "source_video_path" not in row:
                row["source_video_path"] = run_to_path[run_id]

        # Group rows by source video and sort by timestamp (handle ranges)
        by_video = defaultdict(list)
        for row in rows:
            run_id = row.get("run_id")
            vid = row.get("source_video_path")
            if run_id and vid:
                by_video[vid].append(row)

        # Sort rows: use timestamp_seconds if available, otherwise start_timestamp
        for vid in by_video:
            by_video[vid].sort(key=lambda r: r.get("timestamp_seconds") or r.get("start_timestamp") or 0)

        if not by_video:
            print("No matching frames to stitch.", file=sys.stderr)
            cache_db.close()
            sys.exit(0)

        # Build sections from rows (handle ranges)
        all_sections_by_video = {}
        for vid, frames in by_video.items():
            # If frames are ranges (have frame_indices), each range is a section
            if frames and "frame_indices" in frames[0]:
                all_sections_by_video[vid] = [[r] for r in frames]
            else:
                sections = []
                cur = [frames[0]]
                for i in range(1, len(frames)):
                    prev_ts = frames[i - 1]["timestamp_seconds"]
                    curr_ts = frames[i]["timestamp_seconds"]
                    if curr_ts - prev_ts <= threshold:
                        cur.append(frames[i])
                    else:
                        sections.append(cur)
                        cur = [frames[i]]
                sections.append(cur)
                all_sections_by_video[vid] = sections

        # Determine output paths
        if args.stitch and isinstance(args.stitch, str):
            output_paths = [args.stitch]
        else:
            output_paths = [f"{os.path.splitext(os.path.basename(vid))[0]}_stitched.mp4" for vid in by_video]

        # Stitch each video
        for vid_idx, (vid, sections) in enumerate(all_sections_by_video.items()):
            if not os.path.exists(vid):
                print(f"Warning: video not found: {vid}", file=sys.stderr)
                continue
            cap = cv2.VideoCapture(vid)
            fps = cap.get(cv2.CAP_PROP_FPS)
            if not fps or fps <= 0:
                fps = 25.0
            out_path = output_paths[min(vid_idx, len(output_paths)-1)]
            tmpdir = tempfile.mkdtemp(prefix="scs_stitch_")
            segment_files = []
            try:
                for sec_idx, section in enumerate(sections):
                    img_paths = []
                    for f_idx, row in enumerate(section):
                        if "frame_indices" in row:
                            fi = int(row.get("frame_indices")[0])
                        else:
                            fi = int(row.get("frame_index", 0))

                        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
                        ret, frame = cap.read()
                        if ret:
                            img_path = os.path.join(tmpdir, f"seg_{f_idx:04d}.jpg")
                            cv2.imwrite(img_path, frame)
                            img_paths.append(img_path)
                    if img_paths:
                        seg_file = os.path.join(tmpdir, f"segment_{sec_idx:04d}.mp4")
                        pattern = os.path.join(tmpdir, "seg_%04d.jpg")
                        subprocess.run([
                            "ffmpeg", "-y",
                            "-framerate", str(fps),
                            "-i", pattern,
                            "-c:v", "libx264",
                            "-pix_fmt", "yuv420p",
                            "-r", str(fps),
                            seg_file,
                        ], capture_output=True)
                        if os.path.exists(seg_file):
                            segment_files.append(seg_file)
                    cap.release()
                    cap = cv2.VideoCapture(vid)
                if segment_files:
                    concat_file = os.path.join(tmpdir, "concat.txt")
                    with open(concat_file, "w") as cf:
                        for sf in segment_files:
                            cf.write("file '" + sf + "'\n")
                    subprocess.run([
                        "ffmpeg", "-y",
                        "-f", "concat", "-safe", "0", "-i", concat_file,
                        "-c", "copy", out_path,
                    ], capture_output=True)
                    print(f"Stitched {vid} -> {out_path} ({len(segment_files)} section(s))", file=sys.stderr)
                else:
                    print(f"No frames could be extracted from {vid}", file=sys.stderr)
            finally:
                shutil.rmtree(tmpdir)
        cache_db.close()
        sys.exit(0)

    _print_table(rows)
    cache_db.close()
    sys.exit(0)


def cmd_cache(args):
    """Cache management: list, delete, prune."""
    import datetime

    cache_db = CacheDB(args.cache or "./.cache.db")

    if args.cache_cmd == "list":
        # Build per-run list with optional video filter
        video_filter_path = args.video

        # If --video flag is given, find the video_id by path or content hash
        video_filter_id = None
        if video_filter_path:
            # First try exact path match
            video = cache_db.get_video(video_filter_path)
            if video:
                video_filter_id = video["id"]
                print(f"Matched by path: '{video_filter_path}' -> video_id={video['id']}", file=sys.stderr)
            else:
                # Try content hash match - compute hash of the provided path efficiently
                try:
                    provided_hash = _content_hash(video_filter_path)
                    cur = cache_db.conn.execute(
                        "SELECT id, video_path FROM videos WHERE content_hash = ?",
                        (provided_hash,)
                    )
                    row = cur.fetchone()
                    if row:
                        video_filter_id = row[0]
                        print(f"Matched by hash: '{video_filter_path}' -> video_id={row[0]} (cached path: '{row['video_path']}')", file=sys.stderr)
                        # Hash matches but path differs - update to new path
                        if row["video_path"] != video_filter_path:
                            print(f"Hash match: updating video path '{row['video_path']}' -> '{video_filter_path}'", file=sys.stderr)
                            cache_db.conn.execute(
                                "UPDATE videos SET video_path = ?, updated_at = datetime('now') WHERE id = ?",
                                (video_filter_path, row[0])
                            )
                            cache_db.conn.commit()
                except (FileNotFoundError, IOError):
                    print(f"Warning: cannot read file '{video_filter_path}' to compute hash", file=sys.stderr)

        # If --video was given but could not be resolved to any cached video
        if video_filter_path and video_filter_id is None:
            print(f"Error: no cached video matches '{video_filter_path}'", file=sys.stderr)
            cache_db.close()
            sys.exit(1)

        # Per-run SQL: list all runs with their details
        # We need to filter by video_id if --video was provided
        where_clause = ""
        params = []
        if video_filter_id is not None:
            where_clause = "WHERE ar.video_id = ?"
            params.append(video_filter_id)

        query = f"""
            SELECT ar.id as run_id, v.video_path, ar.model, ar.cache_status, ar.created_at,
                   COUNT(fa.id) as answer_count
            FROM analysis_runs ar
            JOIN videos v ON ar.video_id = v.id
            LEFT JOIN frame_answers fa ON ar.id = fa.run_id
            {where_clause}
            GROUP BY ar.id
            ORDER BY ar.created_at DESC
        """
        cur = cache_db.conn.execute(query, params)
        rows = [dict(r) for r in cur.fetchall()]

        if not rows:
            print("(no cached runs found)", file=sys.stderr)
            cache_db.close()
            sys.exit(0)

        # Format columns for per-run display
        print(f"{_sql_with_params(query, params)}\n", file=sys.stderr)
        _print_table(rows)

        cache_db.close()
        sys.exit(0)

    elif args.cache_cmd == "delete":
        if args.video_path:
            # Accept either run_id (integer) or video path (string)
            str_input = str(args.video_path).strip()
            if str_input.isdigit():
                # Input is a run_id - delete the specific run
                run_id = int(str_input)
                with cache_db.conn:
                    cache_db.conn.execute(
                        "DELETE FROM frame_answers WHERE run_id = ?", (run_id,)
                    )
                    cache_db.conn.execute(
                        "DELETE FROM analysis_runs WHERE id = ?", (run_id,)
                    )
                print(f"Deleted run: {run_id}", file=sys.stderr)
                print(f"{_sql_with_params('DELETE FROM frame_answers WHERE run_id = ?', [run_id])}\n", file=sys.stderr)
                print(f"{_sql_with_params('DELETE FROM analysis_runs WHERE id = ?', [run_id])}\n", file=sys.stderr)
                cache_db.close()
                sys.exit(0)
            else:
                # Input is a video path - resolve to video (by path or content hash) and delete all runs for that video
                video = cache_db.get_video(args.video_path)
                video_id_to_delete = None
                if video:
                    video_id_to_delete = video["id"]
                    print(f"Matched by path: '{args.video_path}' -> video_id={video['id']}", file=sys.stderr)
                else:
                    # Try content hash match
                    try:
                        provided_hash = _content_hash(args.video_path)
                        cur = cache_db.conn.execute(
                            "SELECT id, video_path FROM videos WHERE content_hash = ?",
                            (provided_hash,)
                        )
                        row = cur.fetchone()
                        if row:
                            video_id_to_delete = row[0]
                            print(f"Matched by hash: '{args.video_path}' -> video_id={row[0]} (cached path: '{row['video_path']}')", file=sys.stderr)
                            # Hash matches but path differs - update to new path before delete
                            if row["video_path"] != args.video_path:
                                print(f"Hash match: updating video path '{row['video_path']}' -> '{args.video_path}'", file=sys.stderr)
                                cache_db.conn.execute(
                                    "UPDATE videos SET video_path = ?, updated_at = datetime('now') WHERE id = ?",
                                    (args.video_path, row[0])
                                )
                                cache_db.conn.commit()
                    except (FileNotFoundError, IOError):
                        print(f"Warning: cannot read file '{args.video_path}' to compute hash", file=sys.stderr)
                if video_id_to_delete is None:
                    print(f"Video not found: {args.video_path}", file=sys.stderr)
                    cache_db.close()
                    sys.exit(1)
                with cache_db.conn:
                    # Build the query to show what we are deleting
                    sql = "DELETE FROM analysis_runs WHERE video_id = ?; DELETE FROM videos WHERE id = ?"
                    print(f"{_sql_with_params(sql, [video_id_to_delete, video_id_to_delete])
}", file=sys.stderr)
                    cache_db.conn.execute(
                        "DELETE FROM analysis_runs WHERE video_id = ?", (video_id_to_delete,)
                    )
                    cache_db.conn.execute(
                        "DELETE FROM videos WHERE id = ?", (video_id_to_delete,)
                    )
                print(f"Deleted: {args.video_path}", file=sys.stderr)
                cache_db.close()
                sys.exit(0)

    elif args.cache_cmd == "prune":
        cutoff = (datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=args.days)).isoformat()
        cur = cache_db.conn.execute(
            "SELECT id, video_path FROM videos WHERE created_at < ?", (cutoff,)
        )
        to_delete = cur.fetchall()
        if not to_delete:
            print(f"No videos older than {args.days} days", file=sys.stderr)
            cache_db.close()
            sys.exit(0)
        deleted = 0
        with cache_db.conn:
            for v in to_delete:
                # Build query to show what we are deleting
                sql_query = "DELETE FROM analysis_runs WHERE video_id = ?"
                print(f"{_sql_with_params(sql_query, [v['id']])
}", file=sys.stderr)
                cache_db.conn.execute(sql_query, [v['id']])
                cache_db.conn.execute("DELETE FROM videos WHERE id = ?", (v["id"],))
                deleted += 1
        print(f"Pruned {deleted} video(s) older than {args.days} days", file=sys.stderr)
        cache_db.close()
        sys.exit(0)

    elif args.cache_cmd == "clear":
        if not args.yes:
            print("Error: --yes required to confirm full cache clear", file=sys.stderr)
            cache_db.close()
            sys.exit(1)
        with cache_db.conn:
            clear_queries = [
                ("DELETE FROM frame_answers", []),
                ("DELETE FROM analysis_runs", []),
                ("DELETE FROM videos", []),
                ("DELETE FROM cache_stats", []),
            ]
            for sql, params in clear_queries:
                print(f"{_sql_with_params(sql, params)}\n", file=sys.stderr)
                cache_db.conn.execute(sql, params)
        print("Cache cleared", file=sys.stderr)

    elif args.cache_cmd == "clean":
        invalid_videos = cache_db.get_invalid_videos()
        if not invalid_videos:
            print("No invalid cache entries found.", file=sys.stderr)
            cache_db.close()
            sys.exit(0)

        print("The following video paths in the cache are invalid (don't exist or have changed content):", file=sys.stderr)
        for path, reason in invalid_videos:
            print(f"  {path} ({reason})", file=sys.stderr)

        ans = input("Do you want to delete all cache entries for these invalid videos? (y/n): ").strip().lower()
        if ans in ("y", "yes"):
            deleted_count = 0
            for path, reason in invalid_videos:
                cache_db.delete_video_records(path)
                deleted_count += 1
            print(f"Deleted cache for {deleted_count} invalid video(s).", file=sys.stderr)
        else:
            print("Cleanup aborted.", file=sys.stderr)
        
        cache_db.close()
        sys.exit(0)

    cache_db.close()
    sys.exit(1)


def _resolve_paths(paths: list[Path]) -> set[Path]:
    """Normalize a list of paths to a set of absolute, resolved Path objects."""
    return {Path(p).resolve() for p in paths}


def _image_to_b64(image_path: Path, frame_scale: float) -> str:
    """Read an image file, scale it, and return a base-64 encoded JPEG string."""
    try:
        img = Image.open(str(image_path)).convert("RGB")
        scale = frame_scale if frame_scale else 0.1
        new_w = max(1, int(img.width * scale))
        new_h = max(1, int(img.height * scale))
        img = img.resize((new_w, new_h), Image.LANCZOS)
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG", quality=80)
        return base64.b64encode(buffered.getvalue()).decode("utf-8")
    except Exception as e:
        print(f"Error reading image {image_path}: {e}", file=sys.stderr)
        sys.exit(1)


def _get_question_def(cfg: dict, question_name: str):
    """Return the question definition dict for the given name, or None."""
    for q in cfg.get("questions", []):
        if q.get("name") == question_name:
            return q
    return None


def _classify_image_with_ollama(image_path: Path, question_name: str, cfg: dict) -> float:
    """
    Send a single image to Ollama and return the top probability for the given question.
    """
    question_def = _get_question_def(cfg, question_name)
    if question_def is None:
        print(f"Error: question '{question_name}' not found in config.", file=sys.stderr)
        sys.exit(1)

    # build questions payload (same as in cmd_analyze)
    questions_payload = {}
    for q in cfg.get("questions", []):
        questions_payload[q["name"]] = {
            "type": "choice",
            "instructions": q.get("instructions", ""),
            "criteria": q.get("criteria", {}),
        }

    # determine model and state
    model = cfg.get("ollama_decision_model", "typesafe/jev-1.14")
    state = cfg.get("ollama_state_template", "")
    if not state:
        state = ""

    # resolve Ollama endpoint
    endpoint = resolve_ollama_endpoint(cfg.get("ollama_host", "localhost"))

    # encode image
    b64 = _image_to_b64(image_path, cfg.get("ollama_frame_scale", 0.1))

    # build payload
    payload = {
        "model": model,
        "state": state,
        "images": [b64],
        "questions": questions_payload,
    }

    # send request
    try:
        req = urllib.request.Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=cfg.get("ollama_timeout", 60)) as response:
            res_json = json.loads(response.read().decode("utf-8"))
    except Exception as e:
        print(f"Error calling Ollama: {e}", file=sys.stderr)
        sys.exit(1)

    # extract top probability for the requested question
    answers = res_json.get("answers", {})
    q_ans = answers.get(question_name, {})
    probs = q_ans.get("probabilities", {}) if isinstance(q_ans, dict) else {}
    if not isinstance(probs, dict):
        probs = {}
    top_prob = 0.0
    for ch, p in probs.items():
        try:
            pv = float(p)
        except (TypeError, ValueError):
            pv = 0.0
        if pv > top_prob:
            top_prob = pv
    return top_prob


def cmd_isearch(args):
    """
    Image-search command.
    * args.image  – Path to reference image
    * args.question – required question name
    * args.video – optional list of video paths to filter results
    * all other args are the same as in cmd_search
    """
    import cv2
    import os
    from collections import defaultdict

    cfg = load_config()

    # Resolve video set if provided
    video_set = None
    if args.video:
        video_set = _resolve_paths(args.video)

    # Open cache DB early to resolve question-ref if needed
    cache_db = CacheDB(args.cache or cfg.get("cache", "./.cache.db"))

    # Resolve question name
    question_name = args.question
    if args.question_ref:
        ref = args.question_ref
        if ref.isdigit():
            row = cache_db.conn.execute(
                "SELECT fa.question_name FROM frame_answers fa WHERE fa.run_id = ? LIMIT 1",
                (int(ref),),
            ).fetchone()
            if row:
                question_name = row["question_name"]
            else:
                print(f"Warning: no answers found for run_id {ref}", file=sys.stderr)
        else:
            # treat as video path – find latest run for that video
            vid_path = str(Path(ref).resolve())
            video = cache_db.get_video(vid_path)
            if video:
                run_row = cache_db.conn.execute(
                    "SELECT ar.id FROM analysis_runs ar WHERE ar.video_id = ? ORDER BY ar.created_at DESC LIMIT 1",
                    (video["id"],),
                ).fetchone()
                if run_row:
                    row = cache_db.conn.execute(
                        "SELECT fa.question_name FROM frame_answers fa WHERE fa.run_id = ? LIMIT 1",
                        (run_row["id"],),
                    ).fetchone()
                    if row:
                        question_name = row["question_name"]
                else:
                    print(f"Warning: no runs found for video {ref}", file=sys.stderr)
            else:
                print(f"Warning: video not found in cache: {ref}", file=sys.stderr)

    # Classify the reference image to get a probability threshold
    threshold = _classify_image_with_ollama(args.image, question_name, cfg)

    # Build SQL for image search (similar to cmd_search but simpler)
    select_cols = "fa.run_id, fa.frame_index, fa.timestamp_seconds, fa.question_name, fa.top_choice, fa.top_probability"
    from_clause = "frame_answers fa JOIN analysis_runs ar1 ON fa.run_id = ar1.id"
    where_clauses = ["fa.question_name = ?", "fa.top_probability >= ?"]
    params = [question_name, threshold]

    # video filter (new --video flag) - with hash-based lookup
    if video_set:
        from_clause += " JOIN videos v ON ar1.video_id = v.id"
        matched_video_ids = []
        for vp in video_set:
            # Try exact path match first
            cur = cache_db.conn.execute(
                "SELECT id, content_hash FROM videos WHERE video_path = ?", (vp,)
            )
            row = cur.fetchone()
            if row:
                matched_video_ids.append(row["id"])
                print(f"Matched by path: '{vp}' -> video_id={row['id']}", file=sys.stderr)
                continue
            # Fallback: compute hash of input and match by content_hash
            try:
                input_hash = _content_hash(vp)
                cur = cache_db.conn.execute(
                    "SELECT id, video_path FROM videos WHERE content_hash = ?", (input_hash,)
                )
                row = cur.fetchone()
                if row:
                    # Hash matches but path differs - update to new path
                    if row["video_path"] != vp:
                        print(f"Hash match: updating video path '{row['video_path']}' -> '{vp}'", file=sys.stderr)
                        cache_db.conn.execute(
                            "UPDATE videos SET video_path = ?, updated_at = datetime('now') WHERE id = ?",
                            (vp, row["id"])
                        )
                        cache_db.conn.commit()
                    matched_video_ids.append(row["id"])
                    continue
            except (FileNotFoundError, IOError):
                pass
            # No match found
            print(f"Warning: no cached video matches '{vp}'", file=sys.stderr)

        if matched_video_ids:
            video_id_placeholders = ','.join(['?'] * len(matched_video_ids))
            where_clauses.append(f"v.id IN ({video_id_placeholders})")
            params.extend(matched_video_ids)
        else:
            # No valid videos - add impossible condition to return empty
            where_clauses.append("1=0")

    # model filter
    if getattr(args, "model", None):
        cur = cache_db.conn.execute(
            "SELECT DISTINCT id FROM analysis_runs WHERE model = ?", (args.model,)
        )
        model_run_ids = [r[0] for r in cur.fetchall()]
        if not model_run_ids:
            print(f"Error: no analysis runs found for model '{args.model}'", file=sys.stderr)
            cache_db.close()
            sys.exit(1)
        where_clauses.append(f"fa.run_id IN ({','.join(['?'] * len(model_run_ids))})")
        params.extend(model_run_ids)

    # run_id filter
    if getattr(args, "run_id", None) is not None:
        where_clauses.append("fa.run_id = ?")
        params.append(args.run_id)

    # top / top-per-video logic (reuse simplified)
    top_n = args.top
    top_per_video = args.top_per_video
    merge_threshold = args.merge_result_threshold

    # If top-per-video given, use a window function (similar to search)
    if top_per_video is not None:
        query_sql = f"""
            WITH ranked AS (
                SELECT fa.run_id, fa.frame_index, fa.timestamp_seconds, fa.question_name, fa.top_choice, fa.top_probability, v.id AS video_id
                FROM {from_clause}
                WHERE {' AND '.join(where_clauses)}
                ROW_NUMBER() OVER (PARTITION BY v.id ORDER BY fa.top_probability DESC) AS rn
            )
            SELECT run_id, frame_index, timestamp_seconds, question_name, top_choice, top_probability
            FROM ranked
            WHERE rn <= ?
            ORDER BY top_probability DESC
        """
        params.append(top_per_video)
        rows = cache_db.execute_sql(query_sql, params)
    else:
        limit_expr = f"LIMIT {top_n}" if top_n is not None else ""
        query_sql = f"""
            SELECT DISTINCT {select_cols}
            FROM {from_clause}
            WHERE {' AND '.join(where_clauses)}
            ORDER BY fa.top_probability DESC
            {limit_expr}
        """
        rows = cache_db.execute_sql(query_sql, params)

    # sort by timestamp ascending
    def _result_timestamp(r):
        ts = r.get("start_timestamp") if r.get("start_timestamp") is not None else r.get("timestamp_seconds")
        return ts if ts is not None else 0

    rows = sorted(rows, key=lambda r: (_result_timestamp(r), r.get("run_id") or 0))

    # optional save-frame
    if args.save_frame:
        from collections import defaultdict
        run_to_path = {}
        for r in cache_db.execute_sql("SELECT ar.id AS run_id, v.video_path FROM analysis_runs ar JOIN videos v ON ar.video_id = v.id"):
            run_to_path[r["run_id"]] = r["video_path"]
        by_video = defaultdict(list)
        for row in rows:
            run_id = row.get("run_id")
            if run_id and run_id in run_to_path:
                by_video[run_to_path[run_id]].append(row)
        os.makedirs(args.save_frame, exist_ok=True)
        saved = 0
        fmt = "jpg"
        quality = 90
        for video_path, frames in by_video.items():
            if not os.path.exists(video_path):
                print(f"Warning: video not found: {video_path}", file=sys.stderr)
                continue
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"Warning: cannot open video: {video_path}", file=sys.stderr)
                continue
            frames_sorted = sorted(frames, key=lambda r: r.get("frame_index", 0))
            for frame_info in frames_sorted:
                fi = frame_info.get("frame_index", 0)
                cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
                ret, frame = cap.read()
                if not ret:
                    continue
                video_name = os.path.splitext(os.path.basename(video_path))[0]
                q_name = frame_info.get("question_name", "frame")
                choice = frame_info.get("top_choice", "unknown")
                ts = frame_info.get("timestamp_seconds") or frame_info.get("start_timestamp") or 0
                fname = f"{video_name}_f{fi:06d}_{ts:.1f}s_{q_name}_{choice}.{fmt}"
                out_path = os.path.join(args.save_frame, fname)
                if fmt == "jpg":
                    cv2.imwrite(out_path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
                else:
                    cv2.imwrite(out_path, frame)
                saved += 1
            cap.release()
        print(f"Extracted {saved} frame(s) to {args.save_frame}", file=sys.stderr)

    # print table
    if rows:
        _print_table(rows)
    else:
        print("(no rows)")

    cache_db.close()
    sys.exit(0)


def main():
    parser = argparse.ArgumentParser(
        prog="scs",
        description="Semantic video frame classification using an Ollama decision model."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ---- analyze ----
    p_classify = sub.add_parser("analyze", help="Analyze video frames using Ollama choice-question model")
    p_classify.add_argument("video_paths", nargs="+", help="One or more video files, directories, or glob patterns (e.g., video1.mp4 video2/*.mp4)")
    p_classify.add_argument("--questions", "-qs", type=str, default=None, help="JSON array of questions: [{\"name\":..., \"instructions\":..., \"criteria\":{...}}, ...]")
    p_classify.add_argument("--decision-model-sample-rate", "-dmsr", type=int, default=None, help="Frame sampling interval in seconds (default: 3)")
    p_classify.add_argument("--ollama-host", "-oh", default=None, help="Ollama server address (default: 192.168.1.190). Falls back to localhost if unreachable")
    p_classify.add_argument("--ollama-frame-scale", "-ofs", type=float, default=None, help="Scale factor for frames sent to Ollama (0.0-1.0, default: 0.1). Lower = faster")
    p_classify.add_argument("--ollama-model", "-dm", default=None, help="Ollama model name (default: typesafe/jev-1.14)")
    p_classify.add_argument("--ollama-timeout", "-ot", type=int, default=None, help="Ollama request timeout in seconds (default: 60)")
    p_classify.add_argument("--still-frame-threshold", type=float, default=None, help="Similarity (0-1) above which a frame is considered still (default: 0.95)")
    p_classify.add_argument("--same-scene-threshold", type=float, default=None, help="Similarity (0-1) above which frames are considered same scene (default: 0.80)")
    p_classify.add_argument("--recursive", "-R", action="store_true", default=False, help="Recursively scan directories for video files")
    p_classify.add_argument("--cache", type=str, default=None, help="Path to SQLite cache file (default: ./.cache.db)")
    p_classify.add_argument("--ollama-state", "-s", type=str, default=None, help="State context string for Ollama (default: 'sample frame from a video')")
    p_classify.set_defaults(func=cmd_analyze)

    # ---- query ----
    p_query = sub.add_parser("query", help="Query the cache or extract frames based on SQL")
    p_query.add_argument("--cache", type=str, default=None, help="Path to SQLite cache file (default: ./.cache.db)")
    p_query.add_argument("sql", type=str, nargs="?", default="SELECT fa.frame_index, fa.timestamp_seconds, fa.question_name, fa.top_choice, fa.top_probability FROM frame_answers fa JOIN analysis_runs ar ON fa.run_id = ar.id WHERE top_probability > 0.5", help="SQL query to execute")
    p_query.add_argument("--save-frame", type=str, default=None, help="Directory to save extracted frames")
    p_query.set_defaults(func=cmd_query)

    # ---- search ----
    p_search = sub.add_parser("search", help="Search cached frame answers by question")
    p_search.add_argument("video", type=str, nargs="+", help="One or more video files/directories (e.g., video.mp4 video2/*.mp4)")
    p_search.add_argument("--q", dest="q_values", action="append", default=[], help="Filter condition: 'question=choice:prob[:confidence]' (AND) or 'question=choice:prob|conf' (OR between prob/confidence). E.g., activity=yes:0.8:0.7 (prob>=0.8 AND conf>=0.7) or activity=yes:0.8|0.7 (prob>=0.8 OR conf>=0.7). Multiple --q are AND'd.")
    p_search.add_argument("--top", type=int, default=None, help="Limit to top N frames (or ranges) by probability; results are sorted by timestamp ascending.")
    p_search.add_argument(
        "--top-per-video", type=int, default=None,
        help="Limit number of top results per video by probability; results are sorted by timestamp ascending per video (overrides --top if both given)",
    )
    p_search.add_argument("--merge-result-threshold", type=float, default=None, help="Timestamp difference threshold (seconds) to merge adjacent frames with the same score into a single range.")
    p_search.add_argument("--save-frame", type=str, default=None, help="Directory to save extracted frames")
    p_search.add_argument("--cache", type=str, default=None, help="Path to SQLite cache file (default: ./.cache.db)")
    p_search.add_argument("--model", type=str, default=None, help="Filter results by Ollama model name (e.g., clef-flash:9b, d1:3b)")
    p_search.add_argument("--run-id", type=int, default=None, help="Filter results to a specific analysis run ID")
    p_search.add_argument("--stitch", nargs="?", const=True, default=None, metavar="PATH", help="Stitch matched frames into a single video using ffmpeg. Uses --merge-result-threshold (default 1 sec) to group consecutive frames with the same score into ranges. Optional: provide output path (auto-generates if omitted).")
    p_search.set_defaults(func=cmd_search)

    # ---- isearch ----
    p_isearch = sub.add_parser("isearch", help="Search cached frame answers by reference image (single image).")
    p_isearch.add_argument("image", type=Path, help="Path to reference image (jpg, png, webp, bmp …).")
    p_isearch.add_argument("--question", "-Q", type=str, default=None, help="Question name to score the reference image against (optional if --question-ref is given).")
    p_isearch.add_argument("-qr", "--question-ref", type=str, default=None, help="Use the question set from a run_id (integer) or latest run of a video path (string).")
    p_isearch.add_argument("--video", nargs="+", type=Path, default=None, help="One or more video paths to filter results by (space-separated).")
    p_isearch.add_argument("--top", type=int, default=None, help="Limit to top N frames by probability.")
    p_isearch.add_argument("--top-per-video", type=int, default=None, help="Limit number of top results per video by probability.")
    p_isearch.add_argument("--merge-result-threshold", type=float, default=None, help="Timestamp difference threshold (seconds) to merge adjacent frames with same score.")
    p_isearch.add_argument("--save-frame", type=str, default=None, help="Directory to save extracted frames (JPG only).")
    p_isearch.add_argument("--cache", type=str, default=None, help="Path to SQLite cache file (default: ./.cache.db)")
    p_isearch.add_argument("--model", type=str, default=None, help="Filter results by Ollama model name (e.g., clef-flash:9b, d1:3b)")
    p_isearch.add_argument("--run-id", type=int, default=None, help="Filter results to a specific analysis run ID")
    p_isearch.add_argument("--stitch", nargs="?", const=True, default=None, metavar="PATH", help="Stitch matched frames into a single video using ffmpeg.")
    p_isearch.set_defaults(func=cmd_isearch)

    # ---- cache ----
    p_cache = sub.add_parser("cache", help="Cache management: list, delete, prune, clean")
    p_cache.add_argument("--cache", type=str, default=None, help="Path to SQLite cache file (default: ./.cache.db)")
    p_cache.add_argument("cache_cmd", choices=["list", "delete", "prune", "clear", "clean"], help="Cache command")
    p_cache.add_argument("video_path", nargs="?", help="Video path (for delete)")
    p_cache.add_argument("--video", type=str, default=None, help="Video path to filter list (matches by path or content hash)")
    p_cache.add_argument("--limit", type=int, default=1000, help="Limit for list (default: 1000)")
    p_cache.add_argument("--days", type=int, default=30, help="Days threshold for prune (default: 30)")
    p_cache.add_argument("--yes", action="store_true", help="Confirm clear")
    p_cache.set_defaults(func=cmd_cache)

    args = parser.parse_args()

    # Extract cache command args for cmd_cache
    if hasattr(args, 'cache_cmd'):
        args.cache_cmd = args.cache_cmd
    if hasattr(args, 'video_path'):
        args.video_path = args.video_path
    if hasattr(args, 'limit'):
        args.limit = args.limit
    if hasattr(args, 'days'):
        args.days = args.days
    if hasattr(args, 'yes'):
        args.yes = args.yes

    # Extract stitch args
    if hasattr(args, "stitch"):
        args.stitch = args.stitch

    def _global_sigint_handler(signum, frame):
        print("\nInterrupted — saving partial progress before exit...", file=sys.stderr)
        sys.exit(130)

    signal.signal(signal.SIGINT, _global_sigint_handler)
    signal.signal(signal.SIGTERM, _global_sigint_handler)

    args.func(args)


if __name__ == "__main__":
    main()
