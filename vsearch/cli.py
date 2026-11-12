"""Command-line interface for vsearch."""

import argparse
import hashlib
import io
import json
import os
import sys
import urllib.request
import base64

import cv2
import numpy as np

from .engine import VideoSearchEngine, BATCH_SIZE
from .config import load_config, get_config_value, save_config


def cmd_index(args):
    cfg = load_config()
    verbose = bool(getattr(args, "verbose", False))
    model = get_config_value(cfg, "index", "model", getattr(args, "model", None))
    sample_rate = get_config_value(cfg, "index", "sample_rate", getattr(args, "sample_rate", None))
    engine = VideoSearchEngine(model_id=model)
    embeddings, timestamps = engine.extract_and_encode(
        args.video, sample_rate, BATCH_SIZE, verbose=verbose
    )

    # Compute MD5 of the source video for integrity checking
    video_md5 = hashlib.md5()
    with open(args.video, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            video_md5.update(chunk)
    video_md5_hex = video_md5.hexdigest()

    # Auto-generate filename if output is a directory
    output_path = get_config_value(cfg, "index", "output_dir", getattr(args, "output", None))
    video_name = os.path.splitext(os.path.basename(args.video))[0]
    if os.path.isdir(output_path):
        model_name = model.split("/")[-1]  # e.g., clip-vit-base-patch32
        md5_prefix = video_md5_hex[:8]  # first 8 chars
        output_path = os.path.join(output_path, f"{video_name}_{model_name}_{sample_rate}s_{md5_prefix}.npz")
    np.savez_compressed(
        output_path,
        embeddings=embeddings,
        timestamps=timestamps,
        model=model,
        video_path=args.video,
        md5=video_md5_hex,
    )
    if verbose:
        print(f"Cache saved to {output_path} (model: {model})", file=sys.stderr)


def _format_results_text(results):
    """Format results as human-readable text lines."""
    lines = []
    for r in results:
        lines.append(
            f"{r['timestamp_seconds']:.1f}s  {r['timestamp_formatted']}  "
            f"sim={r['score']:.4f}  norm={r['normalized_score']:.4f}"
        )
    return "\n".join(lines)


def _extract_frame_base64(video_path, timestamp):
    """Extract a single frame from video at given timestamp and return base64-encoded JPEG."""
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
    # Encode to JPEG using imencode
    success, encoded = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
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


def _extract_frames_lazy(results, video_path, verbose=False):
    """Yield frames one at a time (lazy generator) to avoid holding all in memory."""
    if not video_path:
        return
    for r in results:
        ts = r["timestamp_seconds"]
        ts_str = f"{int(ts)//3600:02d}{int(ts)%3600//60:02d}{int(ts)%60:02d}"
        frame_b64 = _extract_frame_base64(video_path, ts)
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
        ts_str = f"{ts:.3f}".replace(".", "_")  # e.g., 252.300 -> 252_300
        out_path = os.path.join(output_dir, f"{npz_name}_{ts_str}_{r['score']:.4f}_{r['normalized_score']:.4f}.jpg")
        if _extract_frame_at(video_path, ts, out_path):
            saved += 1
            frame_paths.append(out_path)
            if verbose:
                print(f"  Saved frame at {ts:.1f}s -> {out_path}", file=sys.stderr)
        elif verbose:
            print(f"  Failed to extract frame at {ts:.1f}s", file=sys.stderr)
    return saved, frame_paths



def _validate_results_with_ollama(results, query, frame_b64s, npz_name, model_id="clef-flash", ollama_endpoint="http://localhost:11434/v1/systemone", threshold=0.5):
    """Use Ollama's decision capability to validate frames top-down.

    Frames are checked from highest score to lowest. If a frame fails
    validation (score < threshold), that frame and all lower-score frames are
    discarded. If the highest-score frame fails, all results are discarded.
    """
    try:
        frame_iter = iter(frame_b64s)
        first_b64 = next(frame_iter)
    except StopIteration:
        print("Warning: No frames were extracted. Skipping Ollama validation.", file=sys.stderr)
        return results

    # Process first frame immediately, then remaining via generator
    chain = [first_b64]

    def _chain():
        yield first_b64
        yield from frame_iter

    for i, frame_b64 in enumerate(_chain()):
        payload = {
            "model": model_id,
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
                ollama_endpoint,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req) as response:
                res_json = json.loads(response.read().decode("utf-8"))

            valid_score = res_json["answers"]["is_valid"]["noul"]
            print(f"Ollama decision score for frame {i+1}: {valid_score:.4f}", file=sys.stderr)

            if valid_score < threshold:
                print(f"Warning: Frame {i+1} failed Ollama decision check (score: {valid_score:.4f}, threshold: {threshold:.2f}). Discarding this frame and all lower-score results.", file=sys.stderr)
                # Discard this frame and all lower-score results
                return results[:i]
        except Exception as e:
            print(f"Error validating frame {i+1}: {e}", file=sys.stderr)
            break

    return results


def cmd_search(args):
    cfg = load_config()

    def cfg_val(key, cli_arg):
        return get_config_value(cfg, "search", key, getattr(args, cli_arg, None))

    verbose = bool(getattr(args, "verbose", False))
    top_k = cfg_val("top_k", "top_k")
    raw_floor = cfg_val("raw_floor", "raw_floor")
    norm_threshold = cfg_val("norm_threshold", "norm_threshold")
    json_output = cfg_val("json_output", "json")
    save_frames = cfg_val("save_frames", "save_frames")
    video_path = cfg_val("video", "video")
    validate = cfg_val("decision_model_validate", "decision_model_validate")
    ollama_endpoint = cfg_val("ollama_endpoint", "ollama_endpoint")
    ollama_model = cfg_val("ollama_decision_model", "ollama_model")
    decision_model_threshold = cfg_val("decision_model_threshold", "decision_model_threshold")

    data = np.load(args.cache, allow_pickle=False)
    embeddings = data["embeddings"]
    timestamps = data["timestamps"]
    video_path = data.get("video_path", None)

    # MD5 is mandatory in cache
    if "md5" not in data:
        print(
            f"Error: cache '{args.cache}' is missing required 'md5' field. "
            f"Re-index the video to add integrity metadata.",
            file=sys.stderr,
        )
        sys.exit(1)
    stored_md5 = str(data["md5"].item())

    # If user provided --video, verify MD5 matches the cache's stored MD5
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
    if args.text is not None:
        query = args.text
        results = engine.search_text(query, embeddings, timestamps, top_k, raw_floor, norm_threshold)
    elif args.image is not None:
        query = args.image
        results = engine.search_image(query, embeddings, timestamps, top_k, raw_floor, norm_threshold)
    else:
        print("search requires either --text or --image", file=sys.stderr)
        sys.exit(2)

    # Extract frames in-memory (for validation only) or save to disk
    frame_b64s = []
    frame_paths = []
    if save_frames or validate:
        if not video_path:
            print("--video is required when using --save-frames or --validate", file=sys.stderr)
            sys.exit(2)

        # Extract frames in-memory lazily for validation (no disk writes)
        frame_b64_gen = _extract_frames_lazy(results, video_path, verbose)

        if save_frames:
            os.makedirs(save_frames, exist_ok=True)
            saved_count, frame_paths = _save_matched_frames(results, npz_name, video_path, save_frames, verbose)
            if verbose:
                print(f"Saved {saved_count} frame(s) to {save_frames}", file=sys.stderr)
        if verbose:
            print(f"Extracting frames in memory for validation...", file=sys.stderr)

    # Perform Ollama validation if requested
    if validate and frame_b64_gen:
        results = _validate_results_with_ollama(results, query, frame_b64_gen, npz_name, model_id=ollama_model, ollama_endpoint=ollama_endpoint, threshold=decision_model_threshold)

    # Output to stdout (JSON or text)
    if json_output:
        print(json.dumps({"query": query, "results": results}))
    else:
        if results:
            print(_format_results_text(results))
        else:
            print("No matches found.")


def main():
    parser = argparse.ArgumentParser(prog="vsearch", description="Semantic video timestamp search (offline CLIP).")
    sub = parser.add_subparsers(dest="command", required=True)

    p_index = sub.add_parser("index", help="Extract keyframes and build a .npz cache")
    p_index.add_argument("video", help="Path to input video file")
    p_index.add_argument("--output", "-o", default=None, help="Path to save the output cache archive (.npz). If a directory, auto-generates filename: {video_name}_{model_name}_{sample_rate}_{md5_prefix8}.npz")
    p_index.add_argument("--sample-rate", "-r", type=int, default=None, help="Frame sampling interval in seconds (default: 1)")
    p_index.add_argument("--model", "-m",default=None, help="CLIP model variant (default: openai/clip-vit-base-patch32)")
    p_index.add_argument("--verbose", "-v", action="store_true", default=None, help="Enable verbose progress output")
    p_index.set_defaults(func=cmd_index)

    p_search = sub.add_parser("search", help="Query a pre-computed cache")
    p_search.add_argument("cache", help="Path to pre-computed embeddings cache (.npz)")
    p_search.add_argument("--text", "-t", default=None, help="Natural language search prompt")
    p_search.add_argument("--image", "-i", default=None, help="Path to reference image file")
    p_search.add_argument("--top-k", "-k", type=int, default=None, help="Maximum number of top results to return (default: all)")
    p_search.add_argument("--json", action="store_true", default=None, help="Output results in strict JSON format")
    p_search.add_argument("--raw-floor","-rf", type=float, default=None, help="Absolute noise floor for raw similarity (default: 0.25)")
    p_search.add_argument("--norm-threshold","-nt", type=float, default=None, help="Normalized score threshold (default: 0.75)")
    p_search.add_argument("--verbose", "-v", action="store_true", default=None, help="Enable verbose progress output")
    p_search.add_argument("--save-frames", "-sf", default=None, help="Directory to save matched frame images as {cache_name}_{HHMMSS}_{score}_{norm_score}.jpg")
    p_search.add_argument("--video", "-vid", default=None, help="Path to original video file (required with --save-frames)")
    p_search.add_argument("--decision-model-validate","-dmv", action="store_true", default=None, help="Use local Ollama decision model to validate top results. Results failing validation are discarded from match results.")
    p_search.add_argument("--decision-model-threshold","-dmt", type=float, default=None, help="Ollama decision model threshold (default: 0.5). Results with score < threshold are discarded.")
    p_search.add_argument("--ollama-endpoint","-oe", default=None, help="Ollama API endpoint (default: http://localhost:11434/v1/systemone)")
    p_search.add_argument("--ollama-model","-dm", default=None, help="Ollama model name for validation")
    p_search.set_defaults(func=cmd_search)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
