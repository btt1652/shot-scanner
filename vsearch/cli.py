"""Command-line interface for vsearch (decision-model-only mode).

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
import sys
import tempfile
import urllib.error
import urllib.request
import cv2
import numpy as np
from PIL import Image

from .config import load_config, get_config_value, get_questions
from .cache import CacheDB, _content_hash, _params_hash

DEFAULT_VIDEO_EXTENSIONS = [".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv"]

# Ollama supports at most 4 images per multi-image request, so scene batches are fixed at 4.
MAX_SCENE_BATCH_SIZE = 4


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
    """Try to load cached results from SQLite cache.

    Returns (answers, data) where:
      - Cache hit: answers = frame answer rows, data = {'video_id': ..., 'video_path': ...}
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

        if video.get("content_hash") != content_hash or video.get("params_hash") != params_hash:
            print(f"Cache miss: params or video content changed for {video_path}", file=sys.stderr)
            return None, None

        answers = cache_db.get_answers(video["id"])
        if not answers:
            print(f"Cache miss: no answers for {video_path}", file=sys.stderr)
            return None, None

        print(f"Cache hit: {video_path} ({len(answers)} answer rows)", file=sys.stderr)
        return answers, {"video_id": video["id"], "video_path": video_path}
    except Exception as e:
        print(f"Warning: Failed to read cache: {e}", file=sys.stderr)
        return None, None


def _save_cache(cache, video_path, sample_rate, frame_scale, state, questions_json, model, all_results, total_sampled=None, same_scene_threshold=None, still_frame_threshold=None):
    """Save all answer rows to SQLite cache."""
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
            params_hash=params_hash,
            fps=None,
            total_frames=total_sampled,
            stride=sample_rate,
        )

        cache_db.upsert_frame_answers(video_id, all_results)

        # Mark video as complete
        with cache_db.conn:
            cache_db.conn.execute(
                "UPDATE videos SET cache_status = 'complete' WHERE id = ?",
                (video_id,),
            )

        print(f"cache saved {video_path} ({len(all_results)} answer rows)", file=sys.stderr)
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
        normalized.append({
            "name": name,
            "instructions": instructions or "",
            "criteria": criteria,
        })
    return normalized


def _questions_json(questions):
    """Serialize normalized questions to a canonical JSON string (sorted)."""
    return json.dumps(questions, sort_keys=True, ensure_ascii=False)


def _validate_batch_classify(frames_batch, model, state_template, questions, ollama_timeout, ollama_endpoint, batch_id_counter=None):
    """Validate a batch of frames using multiple choice questions.

    frames_batch: list of tuples (frame_index, timestamp_seconds, b64_image, video_path)
    Returns (answer_rows, next_batch_id).
    """
    if not frames_batch:
        return [], 0
    batch_id = (batch_id_counter or 0) + 1
    ts_list = ", ".join(f"{int(ts)//3600:02d}:{int(ts)%3600//60:02d}:{int(ts)%60:02d}" for _, ts, _, _ in frames_batch)
    print(f"send request [{ts_list}]", file=sys.stderr)

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
            probs = q_ans.get("choice", q_ans) if isinstance(q_ans, dict) else {}
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
                "probabilities_json": json.dumps(probs, sort_keys=True, ensure_ascii=False),
                "raw_response": json.dumps(q_ans, ensure_ascii=False) if isinstance(q_ans, dict) else None,
            })
    print(f"response [{ts_list}] batch_id={batch_id} questions={[q['name'] for q in questions]}", file=sys.stderr)
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


def cmd_classify(args):
    """Classify video frames using an Ollama choice-question model."""
    cfg = load_config()

    def cfg_val(key, cli_arg):
        return get_config_value(cfg, "classify", key, getattr(args, cli_arg, None))

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

    # Process each video: check cache first, then classify if miss
    for vp in video_paths:
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
                continue
            if cache_data is not None:
                video_id = cache_data.get("video_id")
                if video_id is not None:
                    cache_db = CacheDB(cache)
                    last_frame = cache_db.execute_sql(
                        "SELECT MAX(frame_index) AS max_idx FROM frame_answers WHERE video_id = ?",
                        (video_id,),
                    )
                    if last_frame and last_frame[0]["max_idx"] is not None:
                        frame_index = last_frame[0]["max_idx"]
                    batch_rows = cache_db.execute_sql(
                        "SELECT MAX(batch_id) AS max_bid FROM frame_answers WHERE video_id = ?",
                        (video_id,),
                    )
                    if batch_rows and batch_rows[0]["max_bid"] is not None:
                        batch_id_counter = batch_rows[0]["max_bid"]
                    print(f"Resuming from cache: frame_index={frame_index}, validated={batch_id_counter} batches", file=sys.stderr)
                    cache_db.close()
                else:
                    print(f"Cache incomplete for {vp} - resuming from frame 0", file=sys.stderr)

        cap = cv2.VideoCapture(vp)
        if not cap.isOpened():
            print(f"Warning: Cannot open video {vp}, skipping", file=sys.stderr)
            continue

        print(f"Processing video: {vp}", file=sys.stderr)
        prev_small_gray = None
        prev_ts = None

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.get(cv2.CAP_PROP_FRAME_COUNT) is not None else None
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps is None or fps <= 0:
            fps = 30.0
        stride = max(1, int(fps * sample_rate))
        total_sampled = int((total_frames + stride - 1) / stride) if total_frames is not None else 0

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
                        total_extracted += 1
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
                    print(f"{vp} (extracted={total_extracted}/{total_sampled}, validated={total_validated})", file=sys.stderr)

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
        print(f"Use: vsearch cache sql \"SELECT * FROM frame_answers WHERE question_name='{questions[0]['name']}'\" --cache {cache or './.cache.db'}", file=sys.stderr)
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


def cmd_cache(args):
    """Cache management subcommand.

    Usage:
        vsearch cache list [--cache FILE]
        vsearch cache stats [--cache FILE]
        vsearch cache prune [DAYS] [--cache FILE]
        vsearch cache sql "SELECT ..." [--cache FILE]
    """
    cache_db = CacheDB(args.cache or "./.cache.db")
    try:
        if getattr(args, "list", False):
            _print_table(cache_db.list_videos(limit=args.limit or 50))
        elif getattr(args, "stats", False):
            videos = cache_db.list_videos(limit=10000)
            frame_count = cache_db.execute_sql("SELECT COUNT(*) AS n FROM frame_answers")
            total_answers = frame_count[0]["n"] if frame_count else 0
            print(f"Cache file: {args.cache or './.cache.db'}")
            print(f"Database:        {cache_db.db_path}")
            print(f"Total videos:    {len(videos)}")
            print(f"Total answers:   {total_answers}")
        elif getattr(args, "prune", None) is not None:
            days = args.prune or 30
            cache_db.prune_old(max_age_days=days)
            cache_db.set_stat("last_prune", str(days))
            print(f"Pruned cache entries older than {days} days.")
        elif getattr(args, "sql", None):
            sql = args.sql
            if sql.lstrip().upper().startswith(("SELECT", "PRAGMA", "WITH", "EXPLAIN")):
                _print_table(cache_db.execute_sql(sql))
            else:
                rowcount = cache_db.execute_sql_command(sql)
                print(f"OK ({rowcount} rows affected)")
        else:
            print(f"Usage: vsearch cache {{list|stats|prune|sql}} [--cache FILE]")
    finally:
        cache_db.close()


def main():
    parser = argparse.ArgumentParser(
        prog="vsearch",
        description="Semantic video frame classification using an Ollama decision model."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_classify = sub.add_parser("classify", help="Classify video frames using Ollama choice-question model")
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
    p_classify.set_defaults(func=cmd_classify)

    # Cache management subcommand
    p_cache = sub.add_parser("cache", help="Manage the classify result cache")
    cache_parent = argparse.ArgumentParser(add_help=False)
    cache_parent.add_argument("--cache", type=str, default=None, help="Path to SQLite cache file (default: ./.cache.db)")
    cache_sub = p_cache.add_subparsers(dest="cache_action", required=True)
    p_list = cache_sub.add_parser("list", parents=[cache_parent], help="List cached videos")
    p_list.add_argument("--limit", type=int, default=50, help="Max number of videos to list (default: 50)")
    p_list.set_defaults(func=cmd_cache, list=True, stats=False, prune=None, sql=None)
    p_stats = cache_sub.add_parser("stats", parents=[cache_parent], help="Show cache statistics")
    p_stats.set_defaults(func=cmd_cache, list=False, stats=True, prune=None, sql=None)
    p_prune = cache_sub.add_parser("prune", parents=[cache_parent], help="Prune old cache entries")
    p_prune.add_argument("prune", type=int, nargs="?", default=30, help="Delete entries older than N days (default: 30)")
    p_prune.set_defaults(func=cmd_cache, list=False, stats=False, prune=None, sql=None)
    p_sql = cache_sub.add_parser("sql", parents=[cache_parent], help="Execute custom SQL query against cache database")
    p_sql.add_argument("sql", type=str, help="SQL query to execute (SELECT for read-only, others for modification)")
    p_sql.set_defaults(func=cmd_cache, list=False, stats=False, prune=None)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
