"""Core engine for vsearch: CLIP-based video keyframe extraction and search."""

import os
import sys
import json
import numpy as np
import cv2
from PIL import Image

import torch
from transformers import CLIPModel, CLIPProcessor

MODEL_ID = "openai/clip-vit-base-patch32"
BATCH_SIZE = 32


class VideoSearchEngine:
    """Encapsulates device, CLIP model, and processor."""

    def __init__(self, model_id=MODEL_ID):
        self.model_id = model_id
        self.device = "cpu"
        self.processor = CLIPProcessor.from_pretrained(model_id)
        self.model = CLIPModel.from_pretrained(model_id).to(self.device)

    def _load_model(self, model_id, verbose=False):
        """Load a different CLIP model variant."""
        from transformers import CLIPModel, CLIPProcessor
        self.processor = CLIPProcessor.from_pretrained(model_id)
        self.model = CLIPModel.from_pretrained(model_id).to(self.device)
        self.model_id = model_id
        if verbose:
            print(f"Loaded model: {model_id}", file=sys.stderr, flush=True)

    def _encode_image(self, img):
        """Encode a single PIL image -> L2-normalized feature vector."""
        inputs = self.processor(images=[img], return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.get_image_features(**inputs)
        vec = outputs.pooler_output.cpu().numpy().flatten().astype(np.float32)
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    def _encode_text(self, text):
        inputs = self.processor(text=[text], return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            outputs = self.model.get_text_features(**inputs)
        vec = outputs.pooler_output.cpu().numpy().flatten().astype(np.float32)
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    def extract_and_encode(self, video_path, sample_rate_sec=1, batch_size=BATCH_SIZE, verbose=False, model_id=None):
        """Extract frames via OpenCV and return L2-normalized image embeddings + timestamps.

        Streams frames one-at-a-time so very long videos don't exhaust memory.
        Progress is written to stderr for easy monitoring (only when verbose=True).
        """
        import sys as _sys
        if model_id is not None:
            self._load_model(model_id, verbose=verbose)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video file: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps is None or fps <= 0:
            fps = 30.0

        total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        if verbose:
            print(f"Indexing {video_path}: {total_video_frames} frames at {fps:.1f} fps, sampling every {sample_rate_sec}s...", file=_sys.stderr, flush=True)

        embeddings_list = []
        timestamps_list = []
        frame_index = 0
        stride = max(1, int(fps * sample_rate_sec))

        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break
            if frame_index % stride == 0:
                rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb)
                ts = frame_index / fps
                embeddings_list.append(self._encode_image(pil_img))
                timestamps_list.append(ts)
            frame_index += 1
            if verbose and frame_index % 1000 == 0:
                pct = (frame_index / total_video_frames * 100) if total_video_frames > 0 else 0
                print(f"  {frame_index}/{total_video_frames} frames ({pct:.1f}%), encoded {len(embeddings_list)} keyframes...", file=_sys.stderr, flush=True)

        cap.release()

        if not embeddings_list:
            if verbose:
                print(f"No frames extracted from {video_path} (total read: {frame_index})", file=_sys.stderr)
            return np.empty((0,)), np.empty((0,))

        embeddings = np.stack(embeddings_list)
        timestamps_arr = np.array(timestamps_list[: len(embeddings)], dtype=np.float64)
        # L2-normalize each frame's embedding to unit length (row-wise).
        # Embeddings are already unit-norm from _encode_image; this is a robustness safeguard.
        if embeddings.ndim == 1:
            norms = np.linalg.norm(embeddings)
            norms = norms if norms > 0 else 1.0
            embeddings = embeddings / norms
        else:
            norms = np.linalg.norm(embeddings, axis=1)
            norms[norms == 0] = 1.0
            embeddings = embeddings / norms
        if verbose:
            print(f"Extracted {len(embeddings)} frames at {sample_rate_sec}s interval", file=_sys.stderr, flush=True)
        return embeddings, timestamps_arr

    def _rank_results(self, query_vec, embeddings, timestamps, top_k, raw_floor, norm_threshold, use_percentile=False, percentile=0.0):
        """Rank frames by raw cosine similarity (absolute, comparable across videos).

        - raw_floor: calibrated minimum cosine similarity to keep (e.g. 0.5).
        - use_percentile / percentile: optionally keep only the top fraction of
          frames by similarity (e.g. use_percentile=True, percentile=0.2 keeps the
          top 20%), independent of the absolute score.

        `norm_threshold` is kept for backwards-compatibility but no longer
        affects ranking; use raw_floor or the percentile cutoff instead.
        """
        raw_similarities = np.dot(embeddings, query_vec)
        order = np.argsort(-raw_similarities)

        # Optional percentile cutoff: keep only the top `percentile` fraction.
        if use_percentile and 0.0 < percentile < 1.0 and len(raw_similarities) > 0:
            cutoff_idx = max(1, int(len(raw_similarities) * percentile))
            cutoff_score = float(raw_similarities[order[cutoff_idx - 1]])
        else:
            cutoff_score = -np.inf

        results = []
        for idx in order:
            raw_score = float(raw_similarities[idx])

            # Calibrated absolute floor; order is descending so we can stop early.
            if raw_score < raw_floor:
                break
            # Optional percentile cutoff.
            if raw_score < cutoff_score:
                break

            ts = float(timestamps[idx])
            results.append({
                "timestamp_seconds": ts,
                "timestamp_formatted": f"{int(ts)//60:02d}:{int(ts)%60:02d}",
                "score": raw_score,
                "normalized_score": raw_score,
            })
            if top_k is not None and top_k >= 0 and len(results) >= top_k:
                break
        return results

    def search_text(self, text_query, embeddings, timestamps, top_k=5, raw_floor=0.25, norm_threshold=0.75, model_id=None, use_percentile=False, percentile=0.0):
        """Encode text query and run ranked similarity matching."""
        if model_id is not None:
            self._load_model(model_id, verbose=False)
        query_vec = self._encode_text(text_query)
        return self._rank_results(query_vec, embeddings, timestamps, top_k, raw_floor, norm_threshold, use_percentile=use_percentile, percentile=percentile)

    def search_image(self, ref_image_path, embeddings, timestamps, top_k=5, raw_floor=0.25, norm_threshold=0.75, model_id=None, use_percentile=False, percentile=0.0):
        """Encode reference image query and run ranked similarity matching."""
        if model_id is not None:
            self._load_model(model_id, verbose=False)
        from PIL import Image as PILImage
        img = PILImage.open(ref_image_path).convert("RGB")
        query_vec = self._encode_image(img)
        return self._rank_results(query_vec, embeddings, timestamps, top_k, raw_floor, norm_threshold, use_percentile=use_percentile, percentile=percentile)
