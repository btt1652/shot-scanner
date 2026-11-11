"""Command-line interface for vsearch."""

import argparse
import json
import os
import sys

import cv2
import numpy as np

from .engine import VideoSearchEngine, BATCH_SIZE


def cmd_index(args):
    verbose = bool(getattr(args, "verbose", False))
    engine = VideoSearchEngine(model_id=args.model)
    embeddings, timestamps = engine.extract_and_encode(
        args.video, args.sample_rate, BATCH_SIZE, verbose=verbose
    )

    # Auto-generate filename if output is a directory
    output_path = args.output
    if os.path.isdir(output_path):
        video_name = os.path.splitext(os.path.basename(args.video))[0]
        model_name = args.model.split("/")[-1]  # e.g., clip-vit-base-patch32
        output_path = os.path.join(output_path, f"{video_name}_{model_name}.npz")

    video_name = os.path.splitext(os.path.basename(args.video))[0]
    np.savez_compressed(
        output_path,
        embeddings=embeddings,
        timestamps=timestamps,
        video_name=video_name,
    )
    if verbose:
        print(f"Cache saved to {output_path}", file=sys.stderr)


def _format_results_text(results):
    """Format results as human-readable text lines."""
    lines = []
    for r in results:
        lines.append(
            f"{r['timestamp_seconds']:.1f}s  {r['timestamp_formatted']}  "
            f"sim={r['similarity_score']:.4f}  norm={r['normalized_score']:.4f}"
        )
    return "\n".join(lines)


def _save_results_to_file(results, output_path, query, video_name):
    """Save results to a text file. If output_path is a directory, auto-generate filename."""
    if os.path.isdir(output_path):
        from datetime import datetime
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(output_path, f"{video_name}_search_results_{ts}.txt")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"Query: {query}\n")
        f.write(f"Video: {video_name}\n")
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


def _save_matched_frames(results, video_path, output_dir, video_name, verbose=False):
    """Extract and save matched frames as JPEG images."""
    os.makedirs(output_dir, exist_ok=True)
    saved = 0
    for r in results:
        ts = r["timestamp_seconds"]
        ts_str = f"{int(ts)//3600:02d}{int(ts)%3600//60:02d}{int(ts)%60:02d}"
        out_path = os.path.join(output_dir, f"{video_name}_{ts_str}.jpg")
        if _extract_frame_at(video_path, ts, out_path):
            saved += 1
            if verbose:
                print(f"  Saved frame at {ts:.1f}s -> {out_path}", file=sys.stderr)
        elif verbose:
            print(f"  Failed to extract frame at {ts:.1f}s", file=sys.stderr)
    return saved


def cmd_search(args):
    verbose = bool(getattr(args, "verbose", False))
    engine = VideoSearchEngine(model_id=args.model)
    data = np.load(args.cache, allow_pickle=False)
    embeddings = data["embeddings"]
    timestamps = data["timestamps"]
    video_name = data.get("video_name", "unknown")

    if args.text is not None:
        query = args.text
        results = engine.search_text(query, embeddings, timestamps, args.top_k, args.raw_floor, args.norm_threshold)
    elif args.image is not None:
        query = args.image
        results = engine.search_image(query, embeddings, timestamps, args.top_k, args.raw_floor, args.norm_threshold)
    else:
        print("search requires either --text or --image", file=sys.stderr)
        sys.exit(2)

    # Save results to text file if requested
    if args.save_results:
        out_path = _save_results_to_file(results, args.save_results, query, video_name)
        if verbose:
            print(f"Results saved to {out_path}", file=sys.stderr)

    # Save matched frames if requested
    if args.save_frames:
        if not args.video:
            print("--video is required when using --save-frames", file=sys.stderr)
            sys.exit(2)
        saved = _save_matched_frames(results, args.video, args.save_frames, video_name, verbose)
        if verbose:
            print(f"Saved {saved} frame(s) to {args.save_frames}", file=sys.stderr)

    # Output to stdout (JSON or text)
    if args.json:
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
    p_index.add_argument("--output", "-o", default=".", help="Path to save the output cache archive (.npz). If a directory, auto-generates filename: {video_name}_{model_name}.npz")
    p_index.add_argument("--sample-rate", "-r", type=int, default=1, help="Frame sampling interval in seconds (default: 1)")
    p_index.add_argument("--model", default="openai/clip-vit-base-patch32", help="CLIP model variant (default: openai/clip-vit-base-patch32)")
    p_index.add_argument("--verbose", "-v", action="store_true", help="Enable verbose progress output")
    p_index.set_defaults(func=cmd_index)

    p_search = sub.add_parser("search", help="Query a pre-computed cache")
    p_search.add_argument("cache", help="Path to pre-computed embeddings cache (.npz)")
    p_search.add_argument("--text", "-t", default=None, help="Natural language search prompt")
    p_search.add_argument("--image", "-i", default=None, help="Path to reference image file")
    p_search.add_argument("--top-k", "-k", type=int, default=-1, help="Maximum number of top results to return (default: all)")
    p_search.add_argument("--json", action="store_true", help="Output results in strict JSON format")
    p_search.add_argument("--raw-floor", type=float, default=0.25, help="Absolute noise floor for raw similarity (default: 0.25)")
    p_search.add_argument("--norm-threshold", type=float, default=0.75, help="Normalized score threshold (default: 0.75)")
    p_search.add_argument("--model", default="openai/clip-vit-base-patch32", help="CLIP model variant (default: openai/clip-vit-base-patch32)")
    p_search.add_argument("--verbose", "-v", action="store_true", help="Enable verbose progress output")
    p_search.add_argument("--save-results", "-s", default=None, help="Save results to text file. If directory, auto-generates {video_name}_search_results_YYYYMMDD_HHMMSS.txt")
    p_search.add_argument("--save-frames", "-f", default=None, help="Directory to save matched frame images as {video_name}_{HHMMSS}.jpg")
    p_search.add_argument("--video", default=None, help="Path to original video file (required with --save-frames)")
    p_search.set_defaults(func=cmd_search)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()