"""Command-line interface for scs (decision-model-only mode).

This CLI classifies video frames using an Ollama or OpenRouter decision model as the sole judge.
No CLIP embeddings or .npz caches are required.

Questions are defined as a JSON array (via CLI or config.toml) and each frame gets
one answer row per question stored in the SQLite cache.
"""
import argparse
import collections
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
from PIL import Image, ImageDraw, ImageFont

from .config import load_config, get_config_value, get_questions
from .cache import CacheDB, _content_hash, _params_hash

DEFAULT_VIDEO_EXTENSIONS = [".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv"]

# Per-model limits for images per request (OpenRouter Decisions API).
# Ollama is limited to 4; OpenRouter limits vary by model.
# Built-in default model image limits — loaded into config._model_limits_builtin.
# Users can override any of these in config.toml under [model_image_limits].
_BUILTIN_MODEL_LIMITS = {
    "perplexity/pplx-decider-v1.3": 8,
    "perplexity/pplx-decider-v1.2": 8,
    "perplexity/pplx-decider-flash-v1.3": 8,
    "perplexity/pplx-decider-flash-v1.2": 8,
    "typesafe/jev-1.14": 4,
    "typesafe/jev-1.13": 4,
    "cloudflare/clef-flash": 4,
    "cloudflare/clef-omni": 4,
    "microsoft/microsoft-decision-1.1": 8,
    "google/gemini-decision-1": 128,
    "heygen/heygen-decision-1": 100,
    "heygen/heygen-decision-1-turbo": 100,
    "liquid/d1.2": 16,
    "liquid/d1.2-ultra": 16,
    "inception/mercury-decide": 16,
    "baseten/groundhog-1": 16,
    "baseten/bison-1": 16,
}


def _draw_result_on_frame(frame, question_name, top_choice, top_probability, probabilities_json):
    """Draw a minimal result overlay on a frame image.

    Shows question name, top choice with score, and all choices with probabilities.
    Returns the modified frame (numpy array).
    """
    try:
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    except Exception:
        return frame

    draw = ImageDraw.Draw(pil_img)

    # Try to use a TrueType font, fall back to default
    font = None
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    if os.path.exists(font_path):
        try:
            font = ImageFont.truetype(font_path, 14)
            font_small = ImageFont.truetype(font_path, 12)
        except Exception:
            font = ImageFont.load_default()
            font_small = ImageFont.load_default()
    else:
        font = ImageFont.load_default()
        font_small = ImageFont.load_default()

    # Parse probabilities
    probs = {}
    if probabilities_json:
        try:
            probs = json.loads(probabilities_json) if isinstance(probabilities_json, str) else probabilities_json
        except (json.JSONDecodeError, TypeError):
            probs = {}

    # Build overlay lines
    lines = []
    lines.append(f"{question_name}")
    if top_choice:
        lines.append(f"> {top_choice} ({top_probability:.0%})")
    for choice, prob in sorted(probs.items(), key=lambda x: x[1], reverse=True):
        if choice == top_choice:
            continue
        lines.append(f"  {choice} ({prob:.0%})")

    # Calculate text block size
    line_heights = []
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        line_heights.append(bbox[3] - bbox[1] + 4)

    total_height = sum(line_heights) + 16
    padding = 8
    max_line_width = max(draw.textbbox((0, 0), line, font=font)[2] for line in lines)
    box_width = max_line_width + padding * 2
    box_height = total_height + padding * 2

    # Position: top-right corner
    x = pil_img.width - box_width - 10
    y = 10

    # Draw semi-transparent background
    bg = Image.new("RGBA", pil_img.size, (0, 0, 0, 0))
    bg_draw = ImageDraw.Draw(bg)
    bg_draw.rectangle(
        [x, y, x + box_width, y + box_height],
        fill=(0, 0, 0, 180),
    )
    pil_img = Image.alpha_composite(pil_img.convert("RGBA"), bg)
    draw = ImageDraw.Draw(pil_img)

    # Draw text lines
    text_y = y + padding
    for i, line in enumerate(lines):
        color = "#00FF00" if i == 1 else "#FFFFFF"  # Green for top choice, white for others
        draw.text((x + padding, text_y), line, fill=color, font=font)
        text_y += line_heights[i]

    # Convert back to numpy array (BGR)
    result = cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)
    return result


def _get_max_batch_size(provider, model, config=None):
    """Return the max images per request for the given provider and model.

    If config is provided and config["max_images_per_request"] > 0, that value
    is used (capped by the model limit). Otherwise the model limit is looked up:
      1. From config.toml [model_image_limits] table (user overrides)
      2. Merged with built-in defaults (_BUILTIN_MODEL_LIMITS)
      3. Falls back to 4 for unknown models or Ollama.
    """
    if provider != "openrouter":
        return 4  # Ollama always 4

    # Get the effective limits map: user overrides + built-in defaults
    builtin_map = dict(_BUILTIN_MODEL_LIMITS)
    if config:
        user_limits = config.get("model_image_limits")
        if isinstance(user_limits, dict):
            builtin_map.update(user_limits)

    # Look up model limit
    model_limit = builtin_map.get(model, 4)

    # Respect explicit CLI/config override if set
    if isinstance(config, dict):
        cfg_override = config.get("max_images_per_request", -1)
        if isinstance(cfg_override, int) and cfg_override > 0:
            model_limit = min(cfg_override, model_limit)

    return model_limit

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


def _validate_batch_classify(frames_batch, model, state_template, questions, ollama_timeout, ollama_endpoint, batch_id_counter=None, history_state=None, provider="ollama", api_key=""):
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

    # Build provider-specific payload
    if provider == "openrouter":
        state_arr = []
        if state_template:
            state_arr.append(state_template)
        if history_state:
            state_arr.append("\n\n" + history_state)
        for img in images:
            state_arr.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img}"}})
        payload = {"model": model, "state": state_arr, "questions": questions_payload}
    else:
        payload = {
            "model": model,
            "state": (state_template or "") + ("\n\n" + history_state if history_state else ""),
            "images": images,
            "questions": questions_payload,
        }

    try:
        req = urllib.request.Request(
            ollama_endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", **({"Authorization": f"Bearer {api_key}"} if provider == "openrouter" and api_key else {})},
        )
        with urllib.request.urlopen(req, timeout=ollama_timeout) as response:
            res_json = json.loads(response.read().decode("utf-8"))
        answers = res_json.get("answers", {})
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        label = "OpenRouter" if provider == "openrouter" else "Ollama"
        print(f"{label} request failed ({e.code}): {body}. Discarding batch.", file=sys.stderr)
        return [], batch_id
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


def _build_decision_payload(provider, model, state, images_b64, questions_payload):
    """Build the decision request payload for the given provider.

    Ollama uses separate 'state', 'images' fields; OpenRouter embeds images in 'state' array.
    """
    if provider == "openrouter":
        state_array = []
        if state:
            state_array.append(state)
        for img in images_b64:
            state_array.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{img}"},
            })
        return {
            "model": model,
            "state": state_array,
            "questions": questions_payload,
        }
    else:
        return {
            "model": model,
            "state": state,
            "images": images_b64,
            "questions": questions_payload,
        }


def _send_decision_request(endpoint, payload, timeout, provider, api_key=""):
    """Send a decision request and return parsed JSON response."""
    headers = {"Content-Type": "application/json"}
    if provider == "openrouter" and api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(endpoint, data=json.dumps(payload).encode("utf-8"), headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def cmd_analyze(args):
    """Classify video frames using an Ollama or OpenRouter decision model."""
    cfg = load_config()

    def cfg_val(key, cli_arg):
        return get_config_value(cfg, key, getattr(args, cli_arg, None))

    # Read config values (removed: top_k, global_k, json_output, decision_model_floor, merge_range, top_choice)
    frame_sample_rate = cfg_val("frame_sample_rate", "frame_sample_rate")
    provider = cfg_val("provider", "provider") or "ollama"
    ollama_endpoint = cfg_val("ollama_host", "ollama_host")
    ollama_model = cfg_val("decision_model", "decision_model")
    frame_scale = cfg_val("frame_scale", "frame_scale")
    recursive = cfg_val("recursive", "recursive")
    state_template = cfg_val("state_template", "state_template")
    cache = cfg_val("cache", "cache")
    still_frame_threshold = cfg_val("still_frame_threshold", "still_frame_threshold") or 0.95
    same_scene_threshold = cfg_val("same_scene_threshold", "same_scene_threshold") or 0.80
    ollama_timeout = cfg_val("ollama_timeout", "ollama_timeout")
    openrouter_api_key = cfg_val("openrouter_api_key", "openrouter_api_key") or os.environ.get("OPENROUTER_API_KEY", "")
    openrouter_model = cfg_val("openrouter_model", "openrouter_model")
    openrouter_endpoint = cfg_val("openrouter_endpoint", "openrouter_endpoint")
    openrouter_timeout = cfg_val("openrouter_timeout", "openrouter_timeout")

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

    # Resolve endpoint based on provider
    if provider == "openrouter":
        endpoint = openrouter_endpoint or "https://openrouter.ai/api/alpha/decisions"
        model = openrouter_model or "perplexity/pplx-decider-v1.3"
        timeout = openrouter_timeout or 60
        api_key = openrouter_api_key
    else:
        endpoint = resolve_ollama_endpoint(ollama_endpoint)
        model = ollama_model if ollama_model else "typesafe/jev-1.14"
        timeout = ollama_timeout
        api_key = ""

    # max_images_per_request > 0 means explicit override from config.toml or CLI; -1 means auto-detect from model.
    max_batch_size = _get_max_batch_size(provider, model, cfg)
    print(f"Batch limit: {max_batch_size} images/request (provider={provider}, model={model})", file=sys.stderr, flush=True)

    print(f"Config: sample_rate={frame_sample_rate}, frame_scale={frame_scale:.2f}", file=sys.stderr)
    print(f"        provider={provider}, model={model}, host={endpoint}, state={state_template!r}, timeout={timeout}", file=sys.stderr)
    print(f"        questions={[q['name'] for q in questions]}", file=sys.stderr)

    sample_rate = frame_sample_rate if frame_sample_rate else 1
    # model already set above based on provider

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

    # History tracking: store last N processed frames' results for context injection
    # Each entry: (timestamp_seconds, frame_index, question_results_dict, was_still=False)
    history_window_size = cfg_val("frame_history", "frame_history") or 5

    # History tracking: store last N processed frames' results for context injection
    # Each entry: (timestamp_seconds, frame_index, entries_list_or_None, was_still)
    #   entries_list = [{"question_name": ..., "top_choice": ..., "top_probability": ..., "raw_response": ...}, ...]
    #   was_still = True means a still_frame was detected (no model call made)
    history_window_size = cfg_val("frame_history", "frame_history") or 5

    def _format_history_context():
        """Build context string of previous frames' results for Ollama state field.
        
        Always uses full JSON format: dumps complete raw response body per question
        including all probabilities and confidence scores for maximum fidelity.
        
        Returns None if no history available.
        """
        if not video_history:
            return None
        
        lines = []
        lines.append(f"Recent context ({len(video_history)} frame(s)):")

        for ts, fi, entries, was_still in video_history:
            ts_formatted = f"{int(ts)//3600:02d}:{int(ts)%3600//60:02d}:{int(ts)%60:02d}"

            if was_still and entries is None:
                lines.append(f"Frame {fi} ({ts_formatted}) [still_frame]: skipped")
            elif entries is None or len(entries) == 0:
                lines.append(f"Frame {fi} ({ts_formatted}) [no_output]:")
            else:
                # Full JSON mode: dump complete raw responses for maximum fidelity
                lines.append(f"Frame {fi} ({ts_formatted}):")
                for e in entries:
                    if not e.get("raw_response"):
                        continue
                    try:
                        resp = json.loads(e["raw_response"])
                    except (TypeError, ValueError):
                        resp = e["raw_response"]
                    lines.append(f'  {e["question_name"]}: {json.dumps(resp)}')
        
        return "\n".join(lines)

    def _flush_batch():
        nonlocal batch, total_validated, batch_id_counter
        if not batch:
            return
        # Build history context from previous processed frames
        history_state = _format_history_context()
        
        out, batch_id_counter = _validate_batch_classify(
            batch, model, state_template, questions, timeout, endpoint, batch_id_counter, history_state, provider=provider, api_key=api_key
        )
        all_results.extend(out)
        total_validated += len(batch)
        
        # Add this batch's results to history tracking
        # Group output rows by frame_index first
        from collections import defaultdict as dd
        rows_by_frame = dd(list)
        for r in out:
            rows_by_frame[r.get("frame_index", 0)].append(r)
        
        for fi, frame_rows in rows_by_frame.items():
            if fi <= 0:
                continue
            ts = frame_rows[0].get("timestamp_seconds", 0)
            # Build list of question entries with probs and raw response
            q_entries = []
            for r in frame_rows:
                q_entries.append({
                    "question_name": r.get("question_name", ""),
                    "top_choice": r.get("top_choice", ""),
                    "top_probability": r.get("top_probability", 0.0),
                    "raw_response": r.get("raw_response"),
                })
            video_history.append((ts, fi, q_entries, False))
        
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
        
        # Reset history tracking for each video
        video_history = collections.deque(maxlen=history_window_size + 1)
        
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
                        # Preload history from cached results (last N frames)
                        # Group cached results by frame to build per-frame question entry lists
                        from collections import defaultdict as dd2
                        frame_entries_map = dd2(list)
                        for r in cache_results:
                            fi = r.get("frame_index", 0)
                            ts = r.get("timestamp_seconds", 0)
                            qe = {"question_name": r.get("question_name", ""),
                                  "top_choice": r.get("top_choice", ""),
                                  "top_probability": r.get("top_probability", 0.0),
                                  "raw_response": r.get("raw_response")}
                            frame_entries_map[(ts, fi)].append(qe)
                        
                        # Build list of (timestamp, frame_index, entries_list, was_still=False) sorted by timestamp
                        all_frames_history = []
                        for (ts, fi), entries in frame_entries_map.items():
                            all_frames_history.append((ts, fi, entries, False))
                        all_frames_history.sort(key=lambda x: x[0])  # sort by timestamp ascending
                        
                        # Take the last window_size entries for history (most recent frames)
                        preload = list(video_history)
                        preload.extend(all_frames_history[-(history_window_size):])
                        video_history.clear()
                        video_history.extend(preload[-history_window_size:])
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
                        # Track still frame in history for context
                        video_history.append((ts, frame_index, None, True))
                    elif similarity >= same_scene_threshold and len(batch) < max_batch_size:
                        print(f"sim check {ts_cur_formatted} <-> {ts_prev_formatted} sim={similarity:.2f} (same scene)", file=sys.stderr)
                        batch.append((frame_index, ts, b64, vp))
                        total_extracted += 1
                        batch_ts = ", ".join(f"{int(t)//3600:02d}:{int(t)%3600//60:02d}:{int(t)%60:02d}" for _, t, _, _ in batch)
                        print(f"add {ts_cur_formatted} to request [{batch_ts}]", file=sys.stderr)
                        if len(batch) >= max_batch_size:
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

                # Draw result overlay on frame
                probs_json = frame_info.get("probabilities_json")
                top_prob = frame_info.get("top_probability", 0.0)
                frame = _draw_result_on_frame(frame, q_name, choice, top_prob, probs_json)

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

    # Load config for default values (CLI flags override)
    cfg = load_config()
    def cfg_val(key, cli_arg):
        return get_config_value(cfg, key, getattr(args, cli_arg, None))

    # Merge threshold defaults from config; <= 0 means disabled
    merge_same_threshold_default = cfg_val("merge_same_threshold", "merge_same_threshold") or 0
    merge_all_threshold_default = cfg_val("merge_threshold", "merge_threshold") or 0

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

    # Parse conditions with Option 1 syntax:
    #   question=choice:prob>=X,conf>=Y       -> AND (both must meet threshold)
    #   question=choice:prob>=X|conf>=Y       -> OR (either meets threshold)
    #   question=choice:prob>=X               -> only prob threshold
    #   question=choice:conf>=Y               -> only confidence threshold
    # Multiple --q flags are AND'd together
    # Example: --q "activity=yes:prob>=0.8|conf>=0.7" --q "scene_type=couple:prob>=0.9,conf>=0.8"
    #   => (activity=yes AND (prob>=0.8 OR conf>=0.7)) AND (scene_type=couple AND prob>=0.9 AND conf>=0.8)
    conditions = []  # list of (qname, choice, field_conditions, use_or)
    # field_conditions: list of (field, op, value) where field in ('prob', 'conf'), op in ('>=', '<=', '>', '<')
    # use_or: True if conditions within this --q are OR'd, False if AND'd
    for q in q_list:
        # Check for grouped syntax FIRST (before splitting on =): (group) || (group) || ...
        if q.strip().startswith("("):
            groups = []
            use_or = False
            remaining = q.strip()
            while remaining:
                if not remaining.startswith("("):
                    print(f"Error: expected '(' at start of group in '{q}'", file=sys.stderr)
                    cache_db.close()
                    sys.exit(1)
                end = remaining.find(")")
                if end == -1:
                    print(f"Error: unmatched parenthesis in '{q}'", file=sys.stderr)
                    cache_db.close()
                    sys.exit(1)
                group = remaining[1:end].strip()
                groups.append(group)
                remaining = remaining[end+1:].strip()
                if remaining.startswith("||"):
                    use_or = True
                    remaining = remaining[2:].strip()
                elif remaining:
                    print(f"Error: expected '||' after group in '{q}'", file=sys.stderr)
                    cache_db.close()
                    sys.exit(1)
            conditions.append(("group", groups, use_or))
            continue

        parts = q.split('=')
        if len(parts) < 2:
            print(f"Error: invalid --q format '{q}'. Expected 'question=choice:field>=value[,field>=value...]' or '(group)||(group)'.", file=sys.stderr)
            cache_db.close()
            sys.exit(1)
        qname = parts[0].strip()
        rest = parts[1].strip()  # e.g., "yes:prob>=0.8|conf>=0.7" or "yes:prob>=0.8,conf>=0.7"

        # First, extract choice (text before first :)
        if ':' not in rest:
            print(f"Error: missing choice in '{q}'. Expected 'choice:prob>=0.8,...'.", file=sys.stderr)
            cache_db.close()
            sys.exit(1)
        compact_choice, field_part = rest.split(':', 1)
        compact_choice = compact_choice.strip().lower()

        
        # Now parse field_part for conditions
        # Split by || for OR, then by && for AND
        or_parts = [p.strip() for p in field_part.split('||')]
        if len(or_parts) > 1:
            use_or = True
        else:
            use_or = False
            or_parts = [p.strip() for p in field_part.split('&&')]
        
        field_conditions = []
        import re
        for part in or_parts:
            # Parse field OP value
            # Fields: prob, conf, prob:<choice>, rank:<choice>
            # Examples: prob>=0.8, conf ge 0.7, prob:walking>=0.7, rank:running<=2
            match = re.match(r'^(prob|conf|rank)(?::([a-zA-Z_][a-zA-Z0-9_]*))?\s*(>=|ge)\s*([0-9.]+)$', part, re.IGNORECASE)
            if not match:
                print(f"Error: invalid field condition '{part}'. Expected 'prob>=0.8' or 'prob ge 0.8' (only >= operator supported).", file=sys.stderr)
                cache_db.close()
                sys.exit(1)
            field, field_choice, op, value = match.groups()
            # Normalize word operators to symbols for SQL
            # Only >= (or ge) is supported - normalize to >=
            if op.lower() == 'ge':
                op = '>='
            field_conditions.append((field, field_choice, op, float(value)))
        
        if not field_conditions:
            print(f"Error: no valid field conditions in '{q}'.", file=sys.stderr)
            cache_db.close()
            sys.exit(1)
        
        conditions.append(("compact", (qname, compact_choice, field_conditions, use_or)))

    top_n = int(args.top) if args.top is not None else None

    # Build SQL - we need self-join for multiple question conditions on same frame
    # Use frame_answers joined against itself for each condition
    if not conditions:
        print("Error: at least one --q filter required", file=sys.stderr)
        cache_db.close()
        sys.exit(1)

    # Build query with self-joins for multiple conditions
    # First condition uses fa (main frame_answers), subsequent use fa1, fa2, etc.
    select_cols = "fa.run_id, fa.frame_index, fa.timestamp_seconds, fa.question_name, fa.top_choice, fa.top_probability, fa.confidence, fa.probabilities_json, v.id AS video_id"
    from_clause = "frame_answers fa JOIN analysis_runs ar1 ON fa.run_id = ar1.id"
    where_clauses = []
    params = []

    # Build SQL with grouped and compact syntax conditions
    for cond_item in conditions:
        cond_type = cond_item[0]
        
        if cond_type == "group":
            # Grouped syntax: (group) || (group) || ...
            groups = cond_item[1]
            use_or = cond_item[2]
            group_clauses = []
            for group in groups:
                # Parse each group: "qname=choice:field_condition[&&..."
                if ":" not in group:
                    print(f"Error: missing ':' separator in group '{group}'. Expected 'qname=choice:prob>=0.8,...'.", file=sys.stderr)
                    cache_db.close()
                    sys.exit(1)
                group_prefix, group_field_part = group.split(":", 1)
                group_prefix = group_prefix.strip()
                # Extract qname and choice from prefix (e.g., "activity=yes")
                if "=" not in group_prefix:
                    print(f"Error: expected 'qname=choice' before ':' in group '{group}'.", file=sys.stderr)
                    cache_db.close()
                    sys.exit(1)
                group_qname, group_choice = group_prefix.split("=", 1)
                group_qname = group_qname.strip().lower()
                group_choice = group_choice.strip().lower()
                
                group_or_parts = [p.strip() for p in group_field_part.split("||")]
                if len(group_or_parts) > 1:
                    group_use_or = True
                else:
                    group_use_or = False
                    group_or_parts = [p.strip() for p in group_field_part.split("&&")]
                
                group_field_conditions = []
                import re
                for part in group_or_parts:
                    match = re.match(r"^(prob|conf|rank)(?::([a-zA-Z_][a-zA-Z0-9_]*))?\s*(>=|ge)\s*([0-9.]+)$", part, re.IGNORECASE)
                    if not match:
                        print(f"Error: invalid field condition in group '{part}'. Expected 'prob>=0.8' or 'prob ge 0.8'.", file=sys.stderr)
                        cache_db.close()
                        sys.exit(1)
                    field, choice, op, value = match.groups()
                    if op.lower() == "ge":
                        op = ">="
                    group_field_conditions.append((field, choice, op, float(value)))
                
                if not group_field_conditions:
                    print(f"Error: no valid field conditions in group '{group}'.", file=sys.stderr)
                    cache_db.close()
                    sys.exit(1)
                
                # Collect group-local params so we can prepend qname/choice
                group_local_params = []
                gclauses = []
                for field, field_choice, op, value in group_field_conditions:
                    if field == "prob" and field_choice is None:
                        col = "fa.top_probability"
                    elif field == "conf" and field_choice is None:
                        col = "fa.confidence"
                    elif field == "prob" and field_choice is not None:
                        escaped = field_choice.replace('"', '""')
                        json_path = '$."' + field_choice + '"'
                        col = "json_extract(fa.probabilities_json, ?)"
                        group_local_params.append(json_path)
                    elif field == "rank" and field_choice is not None:
                        escaped = field_choice.replace('"', '""')
                        json_path = '$."' + field_choice + '"'
                        col = "(SELECT COUNT(*) + 1 FROM json_each(fa.probabilities_json) WHERE value > json_extract(fa.probabilities_json, ?))"
                    else:
                        print(f"Error: invalid field/choice combination in group: {field}:{field_choice}", file=sys.stderr)
                        cache_db.close()
                        sys.exit(1)
                    gclauses.append(f"{col} {op} ?")
                    group_local_params.append(value)
                
                # Build field filter
                field_filter = "( " + (" OR " if group_use_or else " AND ").join(gclauses) + " )"
                
                # Add qname/choice FIRST, then group local params
                group_params = [group_qname, group_choice] + group_local_params
                params.extend(group_params)
                group_clause = f"fa.question_name = ? AND fa.top_choice = ? AND {field_filter}"
                group_clauses.append(group_clause)
            
            if use_or:
                combined = " OR ".join(group_clauses)
            else:
                combined = " AND ".join(group_clauses)
            where_clauses.append("( " + combined + " )")
            continue
        else:
            # Compact syntax (legacy) - cond_item is ('compact', (qname, choice, field_conditions, use_or))
            _, (qname, choice, field_conditions, use_or) = cond_item
            params.extend([qname, choice])
            
            clauses = []
            for field, field_choice, op, value in field_conditions:
                if field == "prob" and field_choice is None:
                    col = "fa.top_probability"
                elif field == "conf" and field_choice is None:
                    col = "fa.confidence"
                elif field == "prob" and field_choice is not None:
                    escaped = field_choice.replace('"', '""')
                    json_path = '$."' + escaped + '"'
                    col = "json_extract(fa.probabilities_json, ?)"
                    clauses.append(f"{col} {op} ?")
                    params.append(json_path)
                    params.append(value)
                    continue
                elif field == "rank" and field_choice is not None:
                    escaped = field_choice.replace('"', '""')
                    json_path = '$."' + escaped + '"'
                    col = "(SELECT COUNT(*) + 1 FROM json_each(fa.probabilities_json) WHERE value > json_extract(fa.probabilities_json, ?))"
                    clauses.append(f"{col} {op} ?")
                    params.append(json_path)
                    params.append(value)
                    continue
                else:
                    print(f"Error: invalid field/choice combination: {field}:{field_choice}", file=sys.stderr)
                    cache_db.close()
                    sys.exit(1)
                clauses.append(f"{col} {op} ?")
                params.append(value)
            
            if use_or:
                clause = f"fa.question_name = ? AND fa.top_choice = ? AND ({' OR '.join(clauses)})"
            else:
                clause = f"fa.question_name = ? AND fa.top_choice = ? AND ({' AND '.join(clauses)})"
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
    
    # Merge thresholds: CLI flag overrides config default
    # Config values <= 0 mean disabled; CLI value of 0 explicitly disables
    _cli_same = getattr(args, "merge_same_threshold", None)
    _cli_all = getattr(args, "merge_threshold", None)
    
    # Resolve final values (None from CLI means use config default)
    if _cli_same is not None:
        merge_same_threshold = float(_cli_same) if _cli_same != 0 else 0
    else:
        merge_same_threshold = float(merge_same_threshold_default) if merge_same_threshold_default and merge_same_threshold_default > 0 else 0
    
    if _cli_all is not None:
        merge_threshold = float(_cli_all) if _cli_all != 0 else 0
    else:
        merge_threshold = float(merge_all_threshold_default) if merge_all_threshold_default and merge_all_threshold_default > 0 else 0
    
    # Validate: both modes cannot be enabled simultaneously
    if merge_same_threshold > 0 and merge_threshold > 0:
        print("Error: --merge-same-threshold/-mst and --merge-threshold/-mt are mutually exclusive. Enable only one.", file=sys.stderr)
        cache_db.close()
        sys.exit(1)
    
    # Determine which merge mode is active
    use_merge_same = merge_same_threshold > 0
    use_merge_all = merge_threshold > 0
    use_merge = use_merge_same or use_merge_all

    def _merge_frame_rows_into_ranges(rows, threshold, group_by_score=True):
        """Merge consecutive rows into timestamp ranges.
        
        If group_by_score=True (same-mode): only merge rows with identical probability score.
        If group_by_score=False (all-mode): merge ALL adjacent rows by timestamp proximity.
        """
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
                
                if group_by_score:
                    # Same-mode: merge only when score matches AND within threshold
                    should_merge = (
                        current is not None
                        and prob == current.get("top_probability")
                        and (ts - current.get("end_timestamp")) <= threshold
                    )
                else:
                    # All-mode: merge whenever within threshold regardless of score
                    should_merge = (
                        current is not None
                        and (ts - current.get("end_timestamp")) <= threshold
                    )
                
                if should_merge:
                    current["end_timestamp"] = ts
                    current["frame_indices"].append(fid)
                    # Track best probability seen in this range
                    if prob is not None and (current.get("top_probability") is None or prob > current["top_probability"]):
                        current["top_probability"] = prob
                        current["top_choice"] = r.get("top_choice")
                        current["confidence"] = r.get("confidence")
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

        if use_merge_all:
            # All-mode: merge all adjacent rows by timestamp proximity regardless of score
            results = _merge_frame_rows_into_ranges(rows, merge_threshold, group_by_score=False)
        else:
            # Same-mode: merge only rows with identical probability score
            results = _merge_frame_rows_into_ranges(rows, merge_same_threshold, group_by_score=True)

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

                # Draw result overlay on frame
                probs_json = frame_info.get("probabilities_json")
                top_prob = frame_info.get("top_probability", 0.0)
                frame = _draw_result_on_frame(frame, q_name, choice, top_prob, probs_json)

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
        
        # Merge threshold defaults from config; <= 0 means disabled
        _cfg_same = cfg.get("merge_same_threshold", 0) or 0
        _cfg_all = cfg.get("merge_threshold", 0) or 0
        
        # CLI flag overrides config default
        stitch_merge_all = getattr(args, "merge_threshold", None)
        stitch_merge_same = getattr(args, "merge_same_threshold", None)
        
        if stitch_merge_all is not None:
            threshold_all = float(stitch_merge_all) if stitch_merge_all != 0 else 0
        else:
            threshold_all = float(_cfg_all) if _cfg_all > 0 else 0
        
        if stitch_merge_same is not None:
            threshold_same = float(stitch_merge_same) if stitch_merge_same != 0 else 0
        else:
            threshold_same = float(_cfg_same) if _cfg_same > 0 else 0
        
        # Validate: both modes cannot be enabled simultaneously in stitch either
        if threshold_all > 0 and threshold_same > 0:
            print("Error: --merge-same-threshold/-mst and --merge-threshold/-mt are mutually exclusive.", file=sys.stderr)
            cache_db.close()
            sys.exit(1)
        
        if threshold_all > 0:
            threshold = threshold_all
        elif threshold_same > 0:
            threshold = threshold_same
        else:
            threshold = 1.0  # Default fallback

        # Isolated frame padding time (configurable via config.toml, default 3 sec)
        isolated_frame_time_sec = cfg.get("isolated_frame_time", 3) or 3

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

        # Build sections from rows with isolated-frame-time padding
        all_sections_by_video = _build_stitch_sections(by_video, threshold, isolated_frame_time_sec)

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
                    seg_frame_counter = 0
                    for f_idx, row in enumerate(section):
                        # Get all frame indices from this row (ranges may have multiple)
                        fi_list = row.get("frame_indices", None)
                        if fi_list:
                            # Range: extract each individual frame
                            for fi in fi_list:
                                cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
                                ret, frame = cap.read()
                                if ret:
                                    img_path = os.path.join(tmpdir, f"seg_{seg_frame_counter:04d}.jpg")
                                    cv2.imwrite(img_path, frame)
                                    img_paths.append(img_path)
                                    seg_frame_counter += 1
                        else:
                            # Non-range: single timestamp
                            fi = int(row.get("frame_index", 0))
                            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
                            ret, frame = cap.read()
                            if ret:
                                img_path = os.path.join(tmpdir, f"seg_{seg_frame_counter:04d}.jpg")
                                cv2.imwrite(img_path, frame)
                                img_paths.append(img_path)
                                seg_frame_counter += 1
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

    # Output results
    if getattr(args, "json", False):
        # JSON mode for AI agents / programmatic use
        import json as _json
        
        output = []
        for r in rows:
            is_range = "start_timestamp" in r and "end_timestamp" in r
            
            if is_range:
                entries = {
                    "range": True,
                    "start_timestamp_seconds": round(r.get("start_timestamp", 0), 3),
                    "end_timestamp_seconds": round(r.get("end_timestamp", 0), 3),
                    "duration_seconds": round(r.get("end_timestamp", 0) - r.get("start_timestamp", 0), 3),
                    "frame_count": len(r.get("frame_indices", [])),
                    "question_name": r.get("question_name", ""),
                    "top_choice": r.get("top_choice", ""),
                    "top_probability": round(r.get("top_probability", 0), 6),
                    "confidence": round(r.get("confidence", 0), 6) if r.get("confidence") is not None else None,
                    "frame_indices": r.get("frame_indices", []),
                }
                if r.get("video_id"):
                    entries["video_id"] = r["video_id"]
                if r.get("run_id"):
                    entries["run_id"] = r["run_id"]
                output.append(entries)
            else:
                entries = {
                    "range": False,
                    "timestamp_seconds": round(r.get("timestamp_seconds", 0), 3),
                    "frame_index": r.get("frame_index", 0),
                    "question_name": r.get("question_name", ""),
                    "top_choice": r.get("top_choice", ""),
                    "top_probability": round(r.get("top_probability", 0), 6),
                    "confidence": round(r.get("confidence", 0), 6) if r.get("confidence") is not None else None,
                }
                if r.get("video_id"):
                    entries["video_id"] = r["video_id"]
                if r.get("run_id"):
                    entries["run_id"] = r["run_id"]
                output.append(entries)
        
        print(_json.dumps(output, indent=2, ensure_ascii=False))
    else:
        # Default: ASCII table
        if rows:
            _print_table(rows)
        else:
            print("(no rows)")
    
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


def _get_row_ts(row):
    """Get the primary timestamp from a row (for sorting/grouping)."""
    if "start_timestamp" in row:
        return row["start_timestamp"]
    return row.get("timestamp_seconds", 0)


def _get_row_end_ts(row):
    """Get the end timestamp from a row. Ranges use end_timestamp; non-ranges use same as start."""
    if "end_timestamp" in row:
        return row["end_timestamp"]
    return row.get("timestamp_seconds", 0)


def _build_stitch_sections(by_video, threshold_sec, isolated_padding_sec):
    """Build stitch sections from query results, applying isolated-frame-time padding.
    
    For both range and non-range rows: when an isolated gap is detected between groups,
    pad forward from the group's end timestamp by isolated_padding_sec. If the next
    result falls within that padded window, it merges into the same section.
    
    This prevents isolated moments from being cut off from nearby results during stitching.
    
    Args:
        by_video: dict mapping video_path -> list of row dicts (sorted by ts ascending)
        threshold_sec: max gap for direct grouping (e.g., 1.0 sec)
        isolated_padding_sec: extra seconds to pad after isolated group (e.g., 3 sec)
    
    Returns:
        dict mapping video_path -> list of sections, where each section is a list of row dicts
    """
    all_sections_by_video = {}
    
    for vid, frames in by_video.items():
        if not frames:
            continue
        
        # Normalize frames: sort by timestamp, track whether they're ranges
        has_ranges = any("frame_indices" in f for f in frames)
        sorted_frames = sorted(frames, key=lambda r: _get_row_ts(r))
        
        if not has_ranges:
            # Non-range mode: walk through timestamps with padding logic
            sections = []
            cur_section = [sorted_frames[0]]
            prev_end = _get_row_end_ts(sorted_frames[0])
            was_isolated = False
            
            for i in range(1, len(sorted_frames)):
                curr_frame = sorted_frames[i]
                curr_start = _get_row_ts(curr_frame)
                curr_end = _get_row_end_ts(curr_frame)
                gap = curr_start - prev_end
                
                max_gap_allowed = threshold_sec + isolated_padding_sec
                
                if gap <= threshold_sec:
                    # Close enough — keep merging directly
                    cur_section.append(curr_frame)
                    prev_end = curr_end
                    was_isolated = False
                elif gap < max_gap_allowed:
                    # Within padding zone — isolate but still merge into same section
                    cur_section.append(curr_frame)
                    prev_end = curr_end
                    was_isolated = True
                else:
                    # Truly too far — finalize current, start new section
                    sections.append(cur_section)
                    cur_section = [curr_frame]
                    prev_end = curr_end
                    was_isolated = True
            
            # Don't forget last section
            sections.append(cur_section)
            all_sections_by_video[vid] = sections
        else:
            # Range mode: apply padding between consecutive ranges
            sections = []
            cur_section = [sorted_frames[0]]
            prev_range_end = _get_row_end_ts(sorted_frames[0])
            
            for i in range(1, len(sorted_frames)):
                curr_range = sorted_frames[i]
                curr_start = _get_row_ts(curr_range)
                curr_end = _get_row_end_ts(curr_range)
                gap = curr_start - prev_range_end
                
                max_gap_allowed = threshold_sec + isolated_padding_sec
                
                if gap <= threshold_sec:
                    # Close enough — keep in same section
                    cur_section.append(curr_range)
                    prev_range_end = curr_end
                elif gap < max_gap_allowed:
                    # Within padding zone — merge despite isolation
                    cur_section.append(curr_range)
                    prev_range_end = curr_end
                else:
                    # Too far — isolate this range as its own section
                    sections.append(cur_section)
                    cur_section = [curr_range]
                    prev_range_end = curr_end
            
            sections.append(cur_section)
            all_sections_by_video[vid] = sections
    
    return all_sections_by_video


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
    Send a single image to the configured decision provider (Ollama or OpenRouter)
    and return the top probability for the given question.
    """
    provider = cfg.get("provider", "ollama")
    if provider == "openrouter":
        return _classify_image_with_openrouter(image_path, question_name, cfg)
    return _classify_image_with_ollama_impl(image_path, question_name, cfg)


def _classify_image_with_ollama_impl(image_path: Path, question_name: str, cfg: dict) -> float:
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
    model = cfg.get("decision_model", "typesafe/jev-1.14")
    state = cfg.get("state_template", "")
    if not state:
        state = ""

    # resolve Ollama endpoint
    endpoint = resolve_ollama_endpoint(cfg.get("ollama_host", "localhost"))

    # encode image
    b64 = _image_to_b64(image_path, cfg.get("frame_scale", 0.1))

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


def _classify_image_with_openrouter(image_path: Path, question_name: str, cfg: dict) -> float:
    """
    Send a single image to OpenRouter Decisions API and return the top probability.
    """
    question_def = _get_question_def(cfg, question_name)
    if question_def is None:
        print(f"Error: question '{question_name}' not found in config.", file=sys.stderr)
        sys.exit(1)

    # build questions payload (same structure as Ollama)
    questions_payload = {}
    for q in cfg.get("questions", []):
        questions_payload[q["name"]] = {
            "type": "choice",
            "instructions": q.get("instructions", ""),
            "criteria": q.get("criteria", {}),
        }

    # determine model and state
    model = cfg.get("openrouter_model", "perplexity/pplx-decider-v1.3")
    state = cfg.get("state_template", "")
    if not state:
        state = ""

    # resolve endpoint
    endpoint = cfg.get("openrouter_endpoint", "https://openrouter.ai/api/alpha/decisions")
    api_key = cfg.get("openrouter_api_key", "") or os.environ.get("OPENROUTER_API_KEY", "")
    timeout = cfg.get("openrouter_timeout", 60)

    # encode image
    b64 = _image_to_b64(image_path, cfg.get("frame_scale", 0.1))

    # build OpenRouter payload (images embedded in state array)
    state_arr = []
    if state:
        state_arr.append(state)
    state_arr.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
    payload = {"model": model, "state": state_arr, "questions": questions_payload}

    # send request
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    try:
        req = urllib.request.Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
        )
        with urllib.request.urlopen(req, timeout=timeout) as response:
            res_json = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        print(f"OpenRouter request failed ({e.code}): {body}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error calling OpenRouter: {e}", file=sys.stderr)
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
    def cfg_val(key, cli_arg):
        return get_config_value(cfg, key, getattr(args, cli_arg, None))

    # Merge threshold defaults from config; <= 0 means disabled
    merge_same_threshold_default = cfg_val("merge_same_threshold", "merge_same_threshold") or 0
    merge_all_threshold_default = cfg_val("merge_threshold", "merge_threshold") or 0

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

    # top / top-per-video / merge logic
    top_n = args.top
    top_per_video = args.top_per_video
    
    # Merge thresholds: CLI flag overrides config default
    _cli_same = getattr(args, "merge_same_threshold", None)
    _cli_all = getattr(args, "merge_threshold", None)
    
    # Resolve final values (None from CLI means use config default)
    if _cli_same is not None:
        merge_same_threshold = float(_cli_same) if _cli_same != 0 else 0
    else:
        merge_same_threshold = float(merge_same_threshold_default) if merge_same_threshold_default > 0 else 0
    
    if _cli_all is not None:
        merge_all_threshold = float(_cli_all) if _cli_all != 0 else 0
    else:
        merge_all_threshold = float(merge_all_threshold_default) if merge_all_threshold_default > 0 else 0
    
    # Validate: both modes cannot be enabled simultaneously
    if merge_same_threshold > 0 and merge_all_threshold > 0:
        print("Error: --merge-same-threshold/-mst and --merge-threshold/-mt are mutually exclusive. Enable only one.", file=sys.stderr)
        cache_db.close()
        sys.exit(1)
    
    use_merge_same = merge_same_threshold > 0
    use_merge_all = merge_all_threshold > 0
    use_merge = use_merge_same or use_merge_all

    if use_merge:
        # Fetch all matching rows; limits applied to merged ranges in Python.
        query_sql = f"""
            SELECT DISTINCT {select_cols}
            FROM {from_clause}
            WHERE {' AND '.join(where_clauses)}
            ORDER BY fa.top_probability DESC
        """
        print(f"{_sql_with_params(query_sql, params)}\n", file=sys.stderr)
        rows = cache_db.execute_sql(query_sql, params)

        if use_merge_all:
            results = _merge_frame_rows_into_ranges(rows, merge_all_threshold, group_by_score=False)
        else:
            results = _merge_frame_rows_into_ranges(rows, merge_same_threshold, group_by_score=True)

        if top_n is not None:
            results = sorted(results, key=lambda r: (r.get("top_probability") or 0), reverse=True)[:top_n]

        if top_per_video is not None:
            from collections import defaultdict
            by_run = defaultdict(list)
            for r in results:
                by_run[r.get("run_id")].append(r)
            results_merged = []
            for rid, group in by_run.items():
                group_sorted = sorted(group, key=lambda r: (r.get("top_probability") or 0), reverse=True)
                results_merged.extend(group_sorted[:top_per_video])
            results = results_merged
        rows = results
    elif top_per_video is not None:
        # Use a window function to limit the top N frames per video.
        query_sql = f"""
            WITH ranked AS (
                SELECT fa.run_id, fa.frame_index, fa.timestamp_seconds, fa.question_name, fa.top_choice, fa.top_probability, v.id AS video_id,
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

                # Draw result overlay on frame
                probs_json = frame_info.get("probabilities_json")
                top_prob = frame_info.get("top_probability", 0.0)
                frame = _draw_result_on_frame(frame, q_name, choice, top_prob, probs_json)

                if fmt == "jpg":
                    cv2.imwrite(out_path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
                else:
                    cv2.imwrite(out_path, frame)
                saved += 1
            cap.release()
        print(f"Extracted {saved} frame(s) to {args.save_frame}", file=sys.stderr)


    if getattr(args, "json", False):
        # JSON mode for AI agents / programmatic use
        import json as _json
        
        output = []
        for r in rows:
            is_range = "start_timestamp" in r and "end_timestamp" in r
            
            if is_range:
                entries = {
                    "range": True,
                    "start_timestamp_seconds": round(r.get("start_timestamp", 0), 3),
                    "end_timestamp_seconds": round(r.get("end_timestamp", 0), 3),
                    "duration_seconds": round(r.get("end_timestamp", 0) - r.get("start_timestamp", 0), 3),
                    "frame_count": len(r.get("frame_indices", [])),
                    "question_name": r.get("question_name", ""),
                    "top_choice": r.get("top_choice", ""),
                    "top_probability": round(r.get("top_probability", 0), 6),
                    "confidence": round(r.get("confidence", 0), 6) if r.get("confidence") is not None else None,
                    "frame_indices": r.get("frame_indices", []),
                }
                if r.get("video_id"):
                    entries["video_id"] = r["video_id"]
                if r.get("run_id"):
                    entries["run_id"] = r["run_id"]
                output.append(entries)
            else:
                entries = {
                    "range": False,
                    "timestamp_seconds": round(r.get("timestamp_seconds", 0), 3),
                    "frame_index": r.get("frame_index", 0),
                    "question_name": r.get("question_name", ""),
                    "top_choice": r.get("top_choice", ""),
                    "top_probability": round(r.get("top_probability", 0), 6),
                    "confidence": round(r.get("confidence", 0), 6) if r.get("confidence") is not None else None,
                }
                if r.get("video_id"):
                    entries["video_id"] = r["video_id"]
                if r.get("run_id"):
                    entries["run_id"] = r["run_id"]
                output.append(entries)
        
        print(_json.dumps(output, indent=2, ensure_ascii=False))
    else:
        # Default: ASCII table
        if rows:
            _print_table(rows)
        else:
            print("(no rows)")

    cache_db.close()
    sys.exit(0)


def main():
    parser = argparse.ArgumentParser(
        prog="scs",
        description="Semantic video frame classification using an Ollama or OpenRouter decision model."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ---- analyze ----
    p_classify = sub.add_parser("analyze", help="Analyze video frames using an Ollama or OpenRouter decision model")
    p_classify.add_argument("video_paths", nargs="+", help="One or more video files, directories, or glob patterns (e.g., video1.mp4 video2/*.mp4)")
    p_classify.add_argument("--questions", "-qs", type=str, default=None, help="JSON array of questions: [{\"name\":..., \"instructions\":..., \"criteria\":{...}}, ...]")
    p_classify.add_argument("--frame-sample-rate", "-fsr", type=int, default=None, help="Frame sampling interval in seconds (default: 3)")
    p_classify.add_argument("--ollama-host", "-oh", default=None, help="Ollama server address (default: 192.168.1.190). Falls back to localhost if unreachable")
    p_classify.add_argument("--frame-scale", "-fs", type=float, default=None, help="Scale factor for frames sent to the model (0.0-1.0, default: 0.1). Lower = faster")
    p_classify.add_argument("--decision-model", "-dm", default=None, help="Decision model name (default: typesafe/jev-1.14)")
    p_classify.add_argument("--ollama-timeout", "-ot", type=int, default=None, help="Ollama request timeout in seconds (default: 60)")
    p_classify.add_argument("--still-frame-threshold", "-sft", type=float, default=None, help="Similarity (0-1) above which a frame is considered still (default: 0.95)")
    p_classify.add_argument("--same-scene-threshold", "-sst", type=float, default=None, help="Similarity (0-1) above which frames are considered same scene (default: 0.80)")
    p_classify.add_argument("--recursive", "-R", action="store_true", default=False, help="Recursively scan directories for video files")
    p_classify.add_argument("--cache", "-c", type=str, default=None, help="Path to SQLite cache file (default: ./.cache.db)")
    p_classify.add_argument("--state-template", "-s", type=str, default=None, help="State context string prepended to every request (default: 'sample frame from a video')")
    p_classify.add_argument("--frame-history", "-fh", type=int, default=None, help="Number of previous processed frames to include as context (default: 5). Set to 0 to disable.")

    # ---- OpenRouter options ----
    p_classify.add_argument("--provider", type=str, default=None, help="Decision provider: 'ollama' (default) or 'openrouter'")
    p_classify.add_argument("--openrouter-api-key", type=str, default=None, help="OpenRouter API key (required when provider=openrouter)")
    p_classify.add_argument("--openrouter-model", type=str, default=None, help="OpenRouter model name (default: perplexity/pplx-decider-v1.3)")
    p_classify.add_argument("--openrouter-endpoint", type=str, default=None, help="OpenRouter API endpoint (default: https://openrouter.ai/api/alpha/decisions)")
    p_classify.add_argument("--openrouter-timeout", type=int, default=None, help="OpenRouter request timeout in seconds (default: 60)")
    p_classify.add_argument("--max-images-per-request", "-mipr", type=int, default=None, help="Max images per batch request. >0 overrides auto-detection (default: -1 = auto from model). Capped by model's native limit.")
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
    p_search.add_argument("--q", dest="q_values", action="append", default=[], help="Filter condition. Compact: 'question=choice:field>=val[&&field>=val...]' (AND) or 'field>=val||field>=val' (OR). Grouped: '(question=choice:field>=val)||(question=choice:field>=val)' for complex OR/AND combinations. Groups allow independent qname+choice with nested ||/&& operators. Fields: prob, conf, prob:<choice>, rank:<choice>. Operator: >= only (or 'ge' word). E.g., activity=yes:prob>=0.8&&conf>=0.7, or (activity=yes:prob>=0.8||conf>=0.5)||(scene=couple:prob>=0.9). Word 'ge' avoids shell escaping. Multiple --q flags are AND'd together.")
    p_search.add_argument("--top", "-t", type=int, default=None, help="Limit to top N frames (or ranges) by probability; results are sorted by timestamp ascending.")
    p_search.add_argument(
        "--top-per-video", "-tpv", type=int, default=None,
        help="Limit number of top results per video by probability; results are sorted by timestamp ascending per video (overrides --top if both given)",
    )
    p_search.add_argument("--merge-same-threshold", "-mst", type=float, default=None, help="Timestamp difference threshold (seconds) to merge adjacent frames with the SAME score into a single range. Only merges rows sharing an identical top_probability. Default from config.toml: merge_same_threshold = 0 (disabled). Set > 0 to enable. Can be overridden via CLI.")
    p_search.add_argument("--merge-threshold", "-mt", type=float, default=None, help="Timestamp difference threshold (seconds) to merge ALL adjacent query results into ranges based on timestamp proximity alone (regardless of score differences). Produces continuous time-range blocks from any matching frames. Default from config.toml: merge_threshold = 0 (disabled). Set > 0 to enable. Can be overridden via CLI. Mutually exclusive with -mst.")
    p_search.add_argument("--save-frame", "-sf", type=str, default=None, help="Directory to save extracted frames")
    p_search.add_argument("--cache", "-c", type=str, default=None, help="Path to SQLite cache file (default: ./.cache.db)")
    p_search.add_argument("--model", "-m", type=str, default=None, help="Filter results by decision model name (e.g., clef-flash:9b, d1:3b, perplexity/pplx-decider-v1.3)")
    p_search.add_argument("--run-id", "-ri", type=int, default=None, help="Filter results to a specific analysis run ID")
    p_search.add_argument("--stitch", "-st", nargs="?", const=True, default=None, metavar="PATH", help="Stitch matched frames into a single video using ffmpeg. Default threshold is 1 second unless --merge-same-threshold/-mst or --merge-threshold/-mt is provided.")
    p_search.add_argument("--json", "-j", action="store_true", default=False, help="Output results as JSON array (for AI agents / programmatic use). Overrides table output on stdout. Merged ranges include start/end timestamps and frame lists.")
    p_search.set_defaults(func=cmd_search)

    # ---- isearch ----
    p_isearch = sub.add_parser("isearch", help="Search cached frame answers by reference image (single image).")
    p_isearch.add_argument("image", type=Path, help="Path to reference image (jpg, png, webp, bmp …).")
    p_isearch.add_argument("--question", "-Q", type=str, default=None, help="Question name to score the reference image against (optional if --question-ref is given).")
    p_isearch.add_argument("-qr", "--question-ref", type=str, default=None, help="Use the question set from a run_id (integer) or latest run of a video path (string).")
    p_isearch.add_argument("--video", "-v", nargs="+", type=Path, default=None, help="One or more video paths to filter results by (space-separated).")
    p_isearch.add_argument("--top", "-t", type=int, default=None, help="Limit to top N frames by probability.")
    p_isearch.add_argument("--top-per-video", "-tpv", type=int, default=None, help="Limit number of top results per video by probability.")
    p_isearch.add_argument("--merge-same-threshold", "-mst", type=float, default=None, help="Timestamp difference threshold (seconds) to merge adjacent frames with same score.")
    p_isearch.add_argument("--merge-threshold", "-mt", type=float, default=None, help="Timestamp difference threshold (seconds) to merge ALL adjacent query results into ranges based on timestamp proximity alone (regardless of score differences). Mutually exclusive with -mst. Default from config.toml: merge_threshold = 0 (disabled).")
    p_isearch.add_argument("--save-frame", "-sf", type=str, default=None, help="Directory to save extracted frames (JPG only).")
    p_isearch.add_argument("--cache", "-c", type=str, default=None, help="Path to SQLite cache file (default: ./.cache.db)")
    p_isearch.add_argument("--model", "-m", type=str, default=None, help="Filter results by decision model name (e.g., clef-flash:9b, d1:3b, perplexity/pplx-decider-v1.3)")
    p_isearch.add_argument("--run-id", "-ri", type=int, default=None, help="Filter results to a specific analysis run ID")
    p_isearch.add_argument("--stitch", "-st", nargs="?", const=True, default=None, metavar="PATH", help="Stitch matched frames into a single video using ffmpeg. Default threshold is 1 second unless --merge-same-threshold/-mst or --merge-threshold/-mt is provided.")
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
