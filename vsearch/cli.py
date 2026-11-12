"""Command-line interface for vsearch."""

import argparse
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

    # Auto-generate filename if output is a directory
    output_path = get_config_value(cfg, "index", "output_dir", getattr(args, "output", None))
    video_name = os.path.splitext(os.path.basename(args.video))[0]
    if os.path.isdir(output_path):
        model_name = model.split("/")[-1]  # e.g., clip-vit-base-patch32
        output_path = os.path.join(output_path, f"{video_name}_{model_name}_{sample_rate}s.npz")
    np.savez_compressed(
        output_path,
        embeddings=embeddings,
        timestamps=timestamps,
        model=model,
        video_path=args.video,
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


def _save_results_to_file(results, output_path, query, npz_name):
    """Save results to a text file. If output_path is a directory, auto-generate filename."""
    if os.path.isdir(output_path):
        from datetime import datetime
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(output_path, f"{npz_name}_search_results_{ts}.txt")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"Query: {query}\n")
        f.write(f"Video: {npz_name}\n")
        f.write(f"Results: {len(results)}\n")
        f.write("-" * 60 + "\n")
        f.write(_format_results_text(results))
        f.write("\n")
    return output_path


def _extract_frame_at(video_path, timestamp, output_path):
    """Extract a single frame from video at given timestamp and save as JPEG."""
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
    cv2.imwrite(output_path, frame)
    return True


def _save_matched_frames(results, npz_name, video_path, output_dir, verbose=False):
    """Extract and save matched frames as JPEG images."""
    os.makedirs(output_dir, exist_ok=True)
    saved = 0
    frame_paths = []
    for r in results:
        ts = r["timestamp_seconds"]
        ts_str = f"{int(ts)//3600:02d}{int(ts)%3600//60:02d}{int(ts)%60:02d}"
        out_path = os.path.join(output_dir, f"{npz_name}_{ts_str}_{r['score']:.4f}_{r['normalized_score']:.4f}.jpg")
        if _extract_frame_at(video_path, ts, out_path):
            saved += 1
            frame_paths.append(out_path)
            if verbose:
                print(f"  Saved frame at {ts:.1f}s -> {out_path}", file=sys.stderr)
        elif verbose:
            print(f"  Failed to extract frame at {ts:.1f}s", file=sys.stderr)
    return saved, frame_paths


def _validate_results_with_ollama(results, query, frame_paths, npz_name, model_id="clef-flash", ollama_endpoint="http://localhost:11434/v1/systemone", threshold=0.5):
    """Use Ollama's decision capability to validate frames top-down.

    Frames are checked from highest score to lowest. If a frame fails
    validation (score < threshold), that frame and all lower-score frames are
    discarded. If the highest-score frame fails, all results are discarded.
    When top_k is unlimited (e.g., -1), all extracted frames are validated.
    """
    if not frame_paths:
        print("Warning: No frames were saved. Skipping Ollama validation.", file=sys.stderr)
        return results

    # Check all extracted frames (corresponding to the top-k results)
    # When top_k is unlimited (e.g., -1), all results are validated.
    selected_frames = frame_paths

    # Check all frames top-down (highest score first)
    top_n = len(selected_frames)
    for i, frame_path in enumerate(selected_frames):
        try:
            with open(frame_path, "rb") as f:
                image_b64 = base64.b64encode(f.read()).decode('utf-8')
        except Exception as e:
            print(f"Error reading frame {frame_path}: {e}", file=sys.stderr)
            continue

        payload = {
            "model": model_id,
            "state": f"A user is searching for: {query}",
            "images": [image_b64],
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
            print(f"Ollama decision score for frame {i+1}/{top_n} at {frame_path}: {valid_score:.4f}", file=sys.stderr)

            if valid_score < threshold:
                print(f"Warning: Frame {i+1} failed Ollama decision check (score: {valid_score:.4f}, threshold: {threshold:.2f}). Discarding this frame and all lower-score results.", file=sys.stderr)
                # Discard this frame and all lower-score results
                # Results are sorted by score (highest first), so we keep only
                # results before this index
                return results[:i]
        except Exception as e:
            print(f"Error connecting to Ollama at {ollama_endpoint}: {e}", file=sys.stderr)
            print("Ensure Ollama is running and 'clef-flash' model is pulled.", file=sys.stderr)
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
    save_results = cfg_val("save_results", "save_results")
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

    if video_path is not None and not os.path.exists(video_path):
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

    # Save results to text file if requested
    if save_results:
        out_path = _save_results_to_file(results, save_results, query, npz_name)
        if verbose:
            print(f"Results saved to {out_path}", file=sys.stderr)

    # Save matched frames if requested OR if validation is enabled
    saved_count = 0
    frame_paths = []
    if save_frames or validate:
        if not video_path:
            print("--video is required when using --save-frames or --validate", file=sys.stderr)
            sys.exit(2)
        if not save_frames:
            print("Warning: --validate requires --save-frames to provide a directory for images.", file=sys.stderr)
            if not os.path.exists(save_frames) and save_frames is not None:
                print(f"Directory {save_frames} does not exist.", file=sys.stderr)
                sys.exit(1)
        
        saved_count, frame_paths = _save_matched_frames(results, npz_name, video_path, save_frames, verbose)
        if verbose:
            print(f"Saved {saved_count} frame(s) to {save_frames}", file=sys.stderr)
        if verbose:
            print(f"Saved {saved_count} frame(s) to {save_frames}", file=sys.stderr)

    # Perform Ollama validation if requested
    if validate and frame_paths:
        results = _validate_results_with_ollama(results, query, frame_paths, npz_name, model_id=ollama_model, ollama_endpoint=ollama_endpoint, threshold=decision_model_threshold)

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
    p_index.add_argument("--output", "-o", default=None, help="Path to save the output cache archive (.npz). If a directory, auto-generates filename: {video_name}_{model_name}.npz")
    p_index.add_argument("--sample-rate", "-r", type=int, default=None, help="Frame sampling interval in seconds (default: 1)")
    p_index.add_argument("--model", default=None, help="CLIP model variant (default: openai/clip-vit-base-patch32)")
    p_index.add_argument("--verbose", "-v", action="store_true", default=None, help="Enable verbose progress output")
    p_index.set_defaults(func=cmd_index)

    p_search = sub.add_parser("search", help="Query a pre-computed cache")
    p_search.add_argument("cache", help="Path to pre-computed embeddings cache (.npz)")
    p_search.add_argument("--text", "-t", default=None, help="Natural language search prompt")
    p_search.add_argument("--image", "-i", default=None, help="Path to reference image file")
    p_search.add_argument("--top-k", "-k", type=int, default=None, help="Maximum number of top results to return (default: all)")
    p_search.add_argument("--json", action="store_true", default=None, help="Output results in strict JSON format")
    p_search.add_argument("--raw-floor", type=float, default=None, help="Absolute noise floor for raw similarity (default: 0.25)")
    p_search.add_argument("--norm-threshold", type=float, default=None, help="Normalized score threshold (default: 0.75)")
    p_search.add_argument("--verbose", "-v", action="store_true", default=None, help="Enable verbose progress output")
    p_search.add_argument("--save-results", "-s", default=None, help="Save results to text file. If directory, auto-generates {cache_name}_search_results_YYYYMMDD_HHMMSS.txt")
    p_search.add_argument("--save-frames", "-f", default=None, help="Directory to save matched frame images as {cache_name}_{HHMMSS}_{score}_{norm_score}.jpg")
    p_search.add_argument("--video", default=None, help="Path to original video file (required with --save-frames)")
    p_search.add_argument("--decision-model-validate", action="store_true", default=None, help="Use local Ollama decision model to validate top results. Results failing validation are discarded from match results.")
    p_search.add_argument("--decision-model-threshold", type=float, default=None, help="Ollama decision model threshold (default: 0.5). Results with score < threshold are discarded.")
    p_search.add_argument("--ollama-endpoint", default=None, help="Ollama API endpoint (default: http://localhost:11434/v1/systemone)")
    p_search.add_argument("--ollama-model", default=None, help="Ollama model name for validation (default: clef-flash)")
    p_search.set_defaults(func=cmd_search)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
