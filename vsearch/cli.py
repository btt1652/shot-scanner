"""Command-line interface for vsearch."""

import argparse
import json
import sys

import numpy as np

from .engine import VideoSearchEngine, BATCH_SIZE


def cmd_index(args):
    verbose = bool(getattr(args, "verbose", False))
    engine = VideoSearchEngine()
    embeddings, timestamps = engine.extract_and_encode(
        args.video, args.sample_rate, BATCH_SIZE, verbose=verbose
    )
    np.savez_compressed(args.output, embeddings=embeddings, timestamps=timestamps)
    if verbose:
        print(f"Cache saved to {args.output}", file=sys.stderr)


def cmd_search(args):
    verbose = bool(getattr(args, "verbose", False))
    engine = VideoSearchEngine()
    data = np.load(args.cache, allow_pickle=False)
    embeddings = data["embeddings"]
    timestamps = data["timestamps"]

    if args.text is not None:
        query = args.text
        results = engine.search_text(query, embeddings, timestamps, args.top_k, args.raw_floor, args.norm_threshold)
    elif args.image is not None:
        query = args.image
        results = engine.search_image(query, embeddings, timestamps, args.top_k, args.raw_floor, args.norm_threshold)
    else:
        print("search requires either --text or --image", file=sys.stderr)
        sys.exit(2)

    if args.json:
        print(json.dumps({"query": query, "results": results}))
    else:
        for r in results:
            print(f"{r['timestamp_seconds']:.1f}s  {r['timestamp_formatted']}  sim={r['similarity_score']:.4f}  norm={r['normalized_score']:.4f}")


def main():
    parser = argparse.ArgumentParser(prog="vsearch", description="Semantic video timestamp search (offline CLIP).")
    sub = parser.add_subparsers(dest="command", required=True)

    p_index = sub.add_parser("index", help="Extract keyframes and build a .npz cache")
    p_index.add_argument("--video", "-v", required=True, help="Path to input video file")
    p_index.add_argument("--output", "-o", required=True, help="Path to save the output cache archive (.npz)")
    p_index.add_argument("--sample-rate", "-r", type=int, default=1, help="Frame sampling interval in seconds (default: 1)")
    p_index.add_argument("--verbose", action="store_true", help="Enable verbose progress output")
    p_index.set_defaults(func=cmd_index)

    p_search = sub.add_parser("search", help="Query a pre-computed cache")
    p_search.add_argument("--cache", "-c", required=True, help="Path to pre-computed embeddings cache (.npz)")
    p_search.add_argument("--text", "-t", default=None, help="Natural language search prompt")
    p_search.add_argument("--image", "-i", default=None, help="Path to reference image file")
    p_search.add_argument("--top-k", "-k", type=int, default=5, help="Maximum number of top results (default: 5)")
    p_search.add_argument("--json", action="store_true", help="Output results in strict JSON format")
    p_search.add_argument("--raw-floor", type=float, default=0.15, help="Absolute noise floor for raw similarity (default: 0.15)")
    p_search.add_argument("--norm-threshold", type=float, default=0.75, help="Normalized score threshold (default: 0.75)")
    p_search.add_argument("--verbose", action="store_true", help="Enable verbose progress output")
    p_search.set_defaults(func=cmd_search)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()