# vsearch - Semantic Video Timestamp CLI
#
# A lightweight, fully offline command-line interface (CLI) tool designed to extract
# keyframes from local video files, encode them into vector embeddings using an open-source
# Vision-Language Model (CLIP), and cache them locally. Users or AI agents can then query
# the cached video instantly via text prompts or reference images to retrieve matching exact timestamps.
#
# ## Quick Start
#
# 1. Install dependencies:
#    pip install -r requirements.txt
#
# 2. Index a video file (video path is first positional arg):
#    vsearch index myvideo.mp4 -o cache.npz
#
# 3. Search the cache with text:
#    vsearch search cache.npz -t "a person running"
#
# 4. Search with reference image:
#    vsearch search cache.npz -i reference.jpg
#
# ## Model Variants
#
# The tool supports multiple CLIP model variants via the `--model` flag:
#
# - `openai/clip-vit-base-patch32` (default) - Good balance of speed and accuracy
# - `openai/clip-vit-base-patch16` - Smaller, faster on CPU (~400MB)
# - `openai/clip-vit-large-patch14` - Larger, more accurate (~1.3GB, needs more memory)
#
# Example:
#    vsearch index myvideo.mp4 -o cache.npz --model openai/clip-vit-base-patch16
#
# ## Commands
#
# ### `index`
# Process a local video file, extract keyframes, compute embeddings, and save a compressed archive.
#
# Flags:
#   video       Path to input video file (required)
#   -o, --output OUTPUT    Path to save the output cache archive (.npz) (default: current directory).
#                     If a directory, auto-generates filename: {video_name}_{model_name}.npz
#   -r, --sample-rate SAMPLE_RATE  Frame sampling interval in seconds (default: 1)
#   --model MODEL           CLIP model variant (default: openai/clip-vit-base-patch32)
#   --verbose, -v     Enable verbose progress output
#
# ### `search`
# Query a pre-computed cache using either a text prompt or a reference image path.
#
# Flags:
#   cache       Path to pre-computed embeddings cache (.npz) (required)
#   -t, --text        Natural language search prompt
#   -i, --image       Path to reference image file for visual similarity matching
#   -k, --top-k       Maximum number of top results to return (default: 5)
#   --json            Output results in strict JSON format to stdout for AI agent consumption
#   --raw-floor       Absolute noise floor for raw similarity (default: 0.15)
#   --norm-threshold  Normalized score threshold (default: 0.75)
#   --model           CLIP model variant (default: openai/clip-vit-base-patch32)
#   --verbose, -v     Enable verbose progress output
#
# ## Expected JSON Output Schema (`--json` flag)
#
# When `--json` is specified, the application prints strictly valid JSON to stdout matching this schema:
#
# ```json
# {
#   "query": "string (text prompt or image file path)",
#   "results": [
#     {
#       "timestamp_seconds": 252.0,
#       "timestamp_formatted": "04:12",
#       "similarity_score": 0.3125,
#       "normalized_score": 1.0
#     }
#   ]
# }
# ```
#
# ## Architecture
#
# The tool follows a two-phase lifecycle:
#
# 1. **Indexing phase** (`vsearch index`): Extracts frames at the specified sample rate,
#    encodes each frame with CLIP, L2-normalizes the feature vectors, and saves them
#    into a compressed `.npz` archive alongside the original timestamps.
#
# 2. **Query phase** (`vsearch search`): Loads the cached embeddings, encodes the query
#    (text or image) with the same CLIP model, computes cosine similarity via dot product,
#    applies min-max normalization and dual-threshold filtering (normalized >= 0.75 AND raw >= 0.15),
#    sorts by score descending, and returns the top-k matching timestamps.
#
# ## Target Environment
#
# Optimized for CPU execution (compatible with standard multi-core desktop CPUs, 16GB RAM)
# with optional CUDA support if a GPU is present.