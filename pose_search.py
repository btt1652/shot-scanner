#!/usr/bin/env python3
import cv2
import mediapipe as mp
import argparse
import sys
import os
import json
import base64
from datetime import datetime
from urllib import request
from urllib.error import HTTPError, URLError
import numpy as np

CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pose_search.config.json")

def resize_to_fit(target_img, source_img):
    """Resize source_img to fit within target_img dimensions, maintaining aspect ratio.
    
    Returns resized image with padding to match target dimensions.
    """
    h_target, w_target = target_img.shape[:2]
    h_src, w_src = source_img.shape[:2]
    
    # Calculate scale factor to fit within target
    scale = min(w_target / w_src, h_target / h_src)
    new_w = int(w_src * scale)
    new_h = int(h_src * scale)
    
    # Resize source image
    resized = cv2.resize(source_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # Calculate padding to center the image
    pad_w = w_target - new_w
    pad_h = h_target - new_h
    left_pad = pad_w // 2
    right_pad = pad_w - left_pad
    top_pad = pad_h // 2
    bottom_pad = pad_h - top_pad
    
    # Add padding
    result = cv2.copyMakeBorder(resized, top_pad, bottom_pad, left_pad, right_pad,
                                 cv2.BORDER_CONSTANT, value=(0, 0, 0))
    return result


def draw_torso_annotations(image, results):
    """Draw torso landmarks (shoulders + hips) with green lines and points."""
    annotated = image.copy()
    TORSO_LANDMARKS = [11, 12, 23, 24]
    TORSO_CONNECTIONS = [(11, 12), (11, 23), (12, 24), (23, 24)]

    for idx in TORSO_LANDMARKS:
        lm = results.pose_landmarks.landmark[idx]
        h, w = image.shape[:2]
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(annotated, (cx, cy), 6, (0, 255, 0), -1)

    for start_idx, end_idx in TORSO_CONNECTIONS:
        start_lm = results.pose_landmarks.landmark[start_idx]
        end_lm = results.pose_landmarks.landmark[end_idx]
        h, w = image.shape[:2]
        cv2.line(annotated,
                 (int(start_lm.x * w), int(start_lm.y * h)),
                 (int(end_lm.x * w), int(end_lm.y * h)),
                 (0, 255, 0), 2)

    return annotated


def calculate_pose_score(landmarks1, landmarks2, img1_shape=None, img2_shape=None):
    """Calculate matching score (0-100) using raw relative coordinates.

    Compares poses directly by coordinate position on frame (no normalization).
    """
    points1 = np.array([[lm.x, lm.y] for lm in landmarks1], dtype=np.float32)
    points2 = np.array([[lm.x, lm.y] for lm in landmarks2], dtype=np.float32)

    # Normalize for aspect ratio if image shapes provided
    if img1_shape is not None and img2_shape is not None:
        h1, w1 = img1_shape[:2]
        h2, w2 = img2_shape[:2]
        points1[:, 0] = points1[:, 0] * w1 / max(w1, h1)
        points1[:, 1] = points1[:, 1] * h1 / max(w1, h1)
        points2[:, 0] = points2[:, 0] * w2 / max(w2, h2)
        points2[:, 1] = points2[:, 1] * h2 / max(w2, h2)

    # Direct coordinate comparison (no centering, no shoulder scaling)
    distances = np.linalg.norm(points1 - points2, axis=1)
    mean_dist = np.mean(distances)

    # Convert distance to score (0-100): distance 0 = 100, distance > 1 = 0
    score = max(0, 100 - (mean_dist * 100))
    return min(100, max(0, score))


def config_error(message):
    """Print a fatal configuration error and exit."""
    print(f"Error: {message}")
    print(f"Config: {CONFIG_PATH}")
    sys.exit(1)


class ConfigError(Exception):
    """Raised when the config file is missing or invalid."""

    pass


def load_raw_config():
    """Read pose_search.config.json, raising ConfigError on any problem."""
    if not os.path.isfile(CONFIG_PATH):
        raise ConfigError("config file not found")
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            config = json.load(f)
    except json.JSONDecodeError as exc:
        raise ConfigError(
            f"invalid JSON ({exc.msg} at line {exc.lineno}, column {exc.colno})"
        ) from exc
    except OSError as exc:
        raise ConfigError(f"could not read config file ({exc})") from exc
    if not isinstance(config, dict):
        raise ConfigError("config root must be a JSON object")
    return config


def _require_number(section, key, minimum=None, maximum=None):
    value = section.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ConfigError(f'"{key}" must be a number')
    if minimum is not None and value < minimum:
        raise ConfigError(f'"{key}" must be >= {minimum}')
    if maximum is not None and value > maximum:
        raise ConfigError(f'"{key}" must be <= {maximum}')
    return float(value)


def _require_bool(section, key):
    value = section.get(key)
    if not isinstance(value, bool):
        raise ConfigError(f'"{key}" must be true or false')
    return value


def _require_string(section, key, allow_empty=False):
    value = section.get(key)
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        raise ConfigError(f'"{key}" must be a non-empty string')
    return value


def _optional_string(section, key):
    value = section.get(key)
    if value is not None and (not isinstance(value, str) or not value.strip()):
        raise ConfigError(f'"{key}" must be a non-empty string or null')
    return value


def _defaults_section(config, name, fallback):
    """Return the validated `name` section, or `fallback` if missing/invalid."""
    section = config.get(name)
    if not isinstance(section, dict):
        return dict(fallback)
    try:
        result = {}
        for key, validator in fallback.items():
            validator(section, key, result)
        return result
    except ConfigError:
        return dict(fallback)


def load_video_config():
    """Load and validate the fixed video-mode config file."""
    try:
        config = load_raw_config()
    except ConfigError as exc:
        config_error(str(exc))

    video = config.get("video")
    ollama = config.get("ollama")
    if not isinstance(video, dict):
        config_error('missing or invalid "video" object')
    if not isinstance(ollama, dict):
        config_error('missing or invalid "ollama" object')

    def require_number(section, key, minimum=None, maximum=None):
        value = section.get(key)
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            config_error(f'"{key}" must be a number')
        if minimum is not None and value < minimum:
            config_error(f'"{key}" must be >= {minimum}')
        if maximum is not None and value > maximum:
            config_error(f'"{key}" must be <= {maximum}')
        return float(value)

    def optional_string(section, key):
        value = section.get(key)
        if value is not None and (not isinstance(value, str) or not value.strip()):
            config_error(f'"{key}" must be a non-empty string or null')
        return value

    def require_bool(section, key):
        value = section.get(key)
        if not isinstance(value, bool):
            config_error(f'"{key}" must be true or false')
        return value

    def require_string(section, key, allow_empty=False):
        value = section.get(key)
        if not isinstance(value, str) or (not allow_empty and not value.strip()):
            config_error(f'"{key}" must be a non-empty string')
        return value

    video_defaults = {
        "threshold": require_number(video, "threshold", 0.0, 100.0),
        "skip_sec": require_number(video, "skip_sec", 0.1),
        "scale": require_number(video, "scale", 0.1, 1.0),
        "output": optional_string(video, "output"),
        "confidence": require_number(video, "confidence", 0.0, 1.0),
        "complexity": int(require_number(video, "complexity", 0, 2)),
        "threads": int(require_number(video, "threads", 1, 4)),
        "save_frames": require_bool(video, "save_frames"),
        "output_dir": require_string(video, "output_dir"),
    }

    ollama_enabled = ollama.get("enabled")
    if not isinstance(ollama_enabled, bool):
        config_error('"enabled" must be true or false')

    ollama_config = {"enabled": ollama_enabled}
    if ollama_enabled:
        endpoint = require_string(ollama, "endpoint")
        if not endpoint.startswith(("http://", "https://")):
            config_error('"endpoint" must start with http:// or https://')

        state = require_string(ollama, "state")
        model = require_string(ollama, "model")
        question = ollama.get("question")
        if not isinstance(question, dict):
            config_error('"question" must be an object')
        if question.get("type") != "noul":
            config_error('"question" must use type "noul"')

        ollama_config.update({
            "endpoint": endpoint,
            "model": model,
            "state": state,
            "question": question,
            "threshold": require_number(ollama, "threshold", 0.0, 1.0),
            "timeout_sec": require_number(ollama, "timeout_sec", 0.1),
        })

    return {"video": video_defaults, "ollama": ollama_config}


ANALYZE_DEFAULTS = {"confidence": 0.5, "complexity": 1}
COMPARE_DEFAULTS = {"confidence": 0.5, "complexity": 1, "mirror": False}


def read_analyze_defaults():
    """Read analyze section as defaults; fall back to built-in values."""
    try:
        config = load_raw_config()
    except ConfigError:
        return dict(ANALYZE_DEFAULTS)
    section = config.get("analyze")
    if not isinstance(section, dict):
        return dict(ANALYZE_DEFAULTS)
    try:
        confidence = _require_number(section, "confidence", 0.0, 1.0)
        complexity = int(_require_number(section, "complexity", 0, 2))
    except ConfigError:
        return dict(ANALYZE_DEFAULTS)
    return {"confidence": confidence, "complexity": complexity}


def read_compare_defaults():
    """Read compare section as defaults; fall back to built-in values."""
    try:
        config = load_raw_config()
    except ConfigError:
        return dict(COMPARE_DEFAULTS)
    section = config.get("compare")
    if not isinstance(section, dict):
        return dict(COMPARE_DEFAULTS)
    try:
        confidence = _require_number(section, "confidence", 0.0, 1.0)
        complexity = int(_require_number(section, "complexity", 0, 2))
        mirror = _require_bool(section, "mirror")
    except ConfigError:
        return dict(COMPARE_DEFAULTS)
    return {"confidence": confidence, "complexity": complexity, "mirror": mirror}


def call_ollama_decision(annotated_frame, ollama_config):
    """Return the configured NOUL score and full response for one frame."""
    ok, encoded = cv2.imencode(".jpg", annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise RuntimeError("could not encode annotated frame as JPEG")

    payload = {
        "model": ollama_config["model"],
        "state": ollama_config["state"],
        "questions": {"question": ollama_config["question"]},
        "images": [base64.b64encode(encoded.tobytes()).decode("ascii")],
    }

    body = json.dumps(payload).encode("utf-8")
    req = request.Request(
        ollama_config["endpoint"],
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with request.urlopen(req, timeout=ollama_config["timeout_sec"]) as response:
            response_body = response.read().decode("utf-8")
    except HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")[:500]
        raise RuntimeError(f"Ollama returned HTTP {exc.code}: {details}") from exc
    except URLError as exc:
        raise RuntimeError(f"could not connect to Ollama: {exc.reason}") from exc
    except TimeoutError as exc:
        raise RuntimeError("Ollama request timed out") from exc

    try:
        result = json.loads(response_body)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Ollama returned invalid JSON: {exc.msg}") from exc

    answers = result.get("answers") if isinstance(result, dict) else None
    decision = answers.get("question") if isinstance(answers, dict) else None
    noul = decision.get("noul") if isinstance(decision, dict) else None

    if not isinstance(noul, (int, float)) or isinstance(noul, bool):
        raise RuntimeError(
            'Ollama response did not contain a numeric noul value for "question"'
        )

    return float(noul), result


def build_segments(matches, skip_frames, skip_sec):
    """Group matching frames into stitchable segments.

    Matches are already in chronological order. A match belongs to the current
    segment when it is at most one sampling step after the previous match.
    """
    segments = []
    current = None
    previous_frame = None

    for match in matches:
        is_sequential = (
            current is not None
            and previous_frame is not None
            and match["frame"] - previous_frame <= skip_frames
        )

        if is_sequential:
            current["match_frames"].append(match["frame"])
            current["match_scores"].append(match["score"])
            current["end_sec"] = match["time_sec"] + skip_sec
            current["duration_sec"] = current["end_sec"] - current["start_sec"]
        else:
            current = {
                "start_sec": match["time_sec"],
                "end_sec": match["time_sec"] + skip_sec,
                "duration_sec": skip_sec,
                "match_frames": [match["frame"]],
                "match_scores": [match["score"]],
            }
            segments.append(current)

        previous_frame = match["frame"]

    return segments


def main():
    parser = argparse.ArgumentParser(description="Analyze or compare human pose using MediaPipe.")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Single image mode
    parser_single = subparsers.add_parser("analyze", help="Analyze pose in one image")
    parser_single.add_argument("image", help="Path to the image file")
    parser_single.add_argument("--confidence", type=float, default=None, help="Min detection confidence (config default: 0.5)")
    parser_single.add_argument("--complexity", type=int, default=None, choices=[0, 1, 2], help="Model complexity 0=fast, 1=balanced, 2=accurate (config default: 1)")

    # Compare mode
    parser_compare = subparsers.add_parser("compare", help="Compare pose between two images")
    parser_compare.add_argument("image1", help="First image path")
    parser_compare.add_argument("image2", help="Second image path")
    parser_compare.add_argument("--confidence", type=float, default=None, help="Min detection confidence (config default: 0.5)")
    parser_compare.add_argument("--mirror", action=argparse.BooleanOptionalAction, default=None, help="Mirror image2 before comparison (config default: false)")
    parser_compare.add_argument("--complexity", type=int, default=None, choices=[0, 1, 2], help="Model complexity 0=fast, 1=balanced, 2=accurate (config default: 1)")

    # Video scene finder mode
    parser_video = subparsers.add_parser("video", help="Find timestamps in video matching pose reference image")
    parser_video.add_argument("ref_image", help="Path to reference pose image")
    parser_video.add_argument("video_file", help="Path to video file")
    parser_video.add_argument("--threshold", type=float, default=None, help="Match score threshold (0-100, config default: 90)")
    parser_video.add_argument("--skip-sec", type=float, default=None, help="Seconds between compared frames (config default: 3.0)")
    parser_video.add_argument("--scale", type=float, default=None, help="Resize factor for video frames (0.1-1.0, config default: 1.0)")
    parser_video.add_argument("--output", default=None, help="Output JSON manifest path (default: <video-name>.matches.json in current directory)")
    parser_video.add_argument("--confidence", type=float, default=None, help="Min detection confidence (config default: 0.5)")
    parser_video.add_argument("--complexity", type=int, default=None, choices=[0, 1, 2], help="Model complexity 0=fast, 1=balanced, 2=accurate (config default: 1)")
    parser_video.add_argument("--threads", type=int, default=None, choices=[1, 2, 3, 4], help="Number of processing threads (config default: 1; currently unused)")
    parser_video.add_argument("--save-frames", action=argparse.BooleanOptionalAction, default=None, help="Save matched frames with drawn landmarks (config default: false)")
    parser_video.add_argument("--output-dir", default=None, help="Directory to save matched frames (config default: current folder)")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    video_config = None
    ollama_config = None
    if args.command == "video":
        loaded_config = load_video_config()
        video_config = loaded_config["video"]
        ollama_config = loaded_config["ollama"]
        for key in ("threshold", "skip_sec", "scale", "confidence", "complexity", "threads", "save_frames", "output_dir"):
            if getattr(args, key) is None:
                setattr(args, key, video_config[key])
        if args.output is None:
            args.output = video_config["output"]
    elif args.command == "analyze":
        defaults = read_analyze_defaults()
        for key in ("confidence", "complexity"):
            if getattr(args, key) is None:
                setattr(args, key, defaults[key])
    elif args.command == "compare":
        defaults = read_compare_defaults()
        for key in ("confidence", "complexity", "mirror"):
            if getattr(args, key) is None:
                setattr(args, key, defaults[key])

    # Check for MediaPipe solutions
    if not hasattr(mp, 'solutions'):
        print("Error: 'mediapipe.solutions' was not found.")
        print("Reason: MediaPipe is currently incompatible with Python 3.15.")
        print("Solution: Please use Python 3.10, 3.11, or 3.12.")
        sys.exit(1)

    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils

    # Torso landmark indices: 11=L_SHOULDER, 12=R_SHOULDER, 23=L_HIP, 24=R_HIP
    TORSO_LANDMARKS = [11, 12, 23, 24]
    TORSO_CONNECTIONS = [(11, 12), (11, 23), (12, 24), (23, 24)]

    # Set static_image_mode based on command
    static_image_mode = args.command != "video"

    with mp_pose.Pose(static_image_mode=static_image_mode,
                      min_detection_confidence=args.confidence,
                      model_complexity=args.complexity) as pose:

        if args.command == "compare":
            # Load both images
            img1 = cv2.imread(args.image1)
            img2 = cv2.imread(args.image2)
            if img1 is None:
                print(f"Error: Could not read image {args.image1}")
                sys.exit(1)
            if img2 is None:
                print(f"Error: Could not read image {args.image2}")
                sys.exit(1)

            # Process images (mirror only image2 if flag set)
            if args.mirror:
                img2_proc = cv2.flip(img2, 1)  # flip only image2 on y-axis
                img1_proc = img1
            else:
                img1_proc = img1
                img2_proc = img2

            results1 = pose.process(cv2.cvtColor(img1_proc, cv2.COLOR_BGR2RGB))
            results2 = pose.process(cv2.cvtColor(img2_proc, cv2.COLOR_BGR2RGB))

            if not results1.pose_landmarks or not results2.pose_landmarks:
                print("No pose detected in one or both images.")
                sys.exit(1)

            # Get torso landmarks only
            landmarks1 = [results1.pose_landmarks.landmark[i] for i in TORSO_LANDMARKS]
            landmarks2 = [results2.pose_landmarks.landmark[i] for i in TORSO_LANDMARKS]

            # Calculate score (0-100) based on relative coordinate distance
            score = calculate_pose_score(landmarks1, landmarks2)

            # Log comparison details
            print(f"\n{'='*60}")
            print("POSE COMPARISON DETAILS")
            print(f"{'='*60}")
            print(f"{'Landmark':<20} | {'Image 1 (x,y)':<15} | {'Image 2 (x,y)':<15} | {'Diff':<10}")
            print("-" * 65)
            for i, idx in enumerate(TORSO_LANDMARKS):
                lm1 = landmarks1[i]
                lm2 = landmarks2[i]
                name = mp_pose.PoseLandmark(idx).name
                diff_x = abs(lm1.x - lm2.x)
                diff_y = abs(lm1.y - lm2.y)
                print(f"{name:<20} | ({lm1.x:.3f},{lm1.y:.3f}) | ({lm2.x:.3f},{lm2.y:.3f}) | dx={diff_x:.3f} dy={diff_y:.3f}")
            print(f"{'='*60}")
            print(f"Score: {score:.1f}/100")
            print(f"{'='*60}")

            # Draw annotations on images (use mirrored versions if flag set)
            annotated1 = draw_torso_annotations(img1_proc.copy(), results1)
            annotated2 = draw_torso_annotations(img2_proc.copy(), results2)

            # Default: overlay blend
            overlay = annotated1.copy()
            cv2.addWeighted(annotated2, 0.5, overlay, 0.5, 0, overlay)
            display_image = overlay
            title = "Compare: Overlay (Press any key)"

            # Add score text at top
            score_text = f"MATCH SCORE: {score:.1f} / 100"
            cv2.putText(display_image, score_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
            # Add image labels
            cv2.putText(display_image, args.image1, (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(display_image, args.image2, (10, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            if args.mirror:
                cv2.putText(display_image, "MIRRORED", (10, 130),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

            # Resize display to 80% scale
            h, w = display_image.shape[:2]
            display_image = cv2.resize(display_image, (int(w * 0.8), int(h * 0.8)), interpolation=cv2.INTER_AREA)

            print(f"\n{'='*60}")
            print(f"POSE COMPARISON RESULT")
            print(f"{'='*60}")
            print(f"Image 1: {args.image1}")
            print(f"Image 2: {args.image2}")
            print(f"Mode: Overlay")
            if args.mirror:
                print(f"Mirror: Yes")
            print(f"Score: {score:.1f} / 100")
            print(f"{'='*60}")
            if score >= 95:
                print("Result: EXCELLENT MATCH")
            elif score >= 80:
                print("Result: GOOD MATCH")
            elif score >= 60:
                print("Result: MODERATE MATCH")
            else:
                print("Result: LOW MATCH")

            cv2.imshow(title, display_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        elif args.command == "video":
            # Load reference image pose
            ref_img = cv2.imread(args.ref_image)
            if ref_img is None:
                print(f"Error: Could not read reference image {args.ref_image}")
                sys.exit(1)

            ref_results = pose.process(cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB))
            if not ref_results.pose_landmarks:
                print("No pose detected in reference image.")
                sys.exit(1)

            # Get reference torso landmarks (direct comparison, no normalization)
            ref_landmarks = [ref_results.pose_landmarks.landmark[i] for i in TORSO_LANDMARKS]

            # Open video file
            cap = cv2.VideoCapture(args.video_file)
            if not cap.isOpened():
                print(f"Error: Could not open video file {args.video_file}")
                sys.exit(1)

            # Get video FPS and total frames
            video_fps = cap.get(cv2.CAP_PROP_FPS)
            video_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            video_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            if video_fps <= 0:
                cap.release()
                print(f"Error: Could not determine FPS for video file {args.video_file}")
                sys.exit(1)

            # Calculate frame skip based on skip-sec
            skip_frames = max(1, int(video_fps * args.skip_sec))

            # Prepare output manifest
            video_stem = os.path.splitext(os.path.basename(args.video_file))[0]
            manifest_path = args.output or f"{video_stem}.matches.json"
            matching_entries = []
            samples_compared = 0

            print(f"Processing video: {args.video_file}")
            print(f"Video FPS: {video_fps:.2f}, Skip frames: {skip_frames} (every {args.skip_sec} sec)")
            print(f"Threshold: {args.threshold}, Output: {manifest_path}")
            print("="*60)

            frame_count = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Process every skip_frames frames
                if frame_count % skip_frames == 0:
                    samples_compared += 1
                    # Calculate timestamp
                    timestamp = frame_count / video_fps
                    minutes = int(timestamp // 60)
                    seconds = timestamp % 60

                    # Resize frame for faster processing
                    if args.scale < 1.0:
                        h, w = frame.shape[:2]
                        frame = cv2.resize(frame, (int(w * args.scale), int(h * args.scale)), interpolation=cv2.INTER_AREA)

                    # Process frame
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = pose.process(frame_rgb)

                    # Log frame being processed
                    if results.pose_landmarks:
                        # Get torso landmarks
                        landmarks = [results.pose_landmarks.landmark[i] for i in TORSO_LANDMARKS]

                        # Calculate pose score using direct coordinate comparison (no normalization)
                        points = np.array([[lm.x, lm.y] for lm in landmarks], dtype=np.float32)
                        ref_points = np.array([[lm.x, lm.y] for lm in ref_landmarks], dtype=np.float32)

                        # Direct Euclidean distance between corresponding landmarks
                        distances = np.linalg.norm(points - ref_points, axis=1)
                        mean_dist = np.mean(distances)

                        # Convert distance to score (0-100): distance 0 = 100, distance > 1 = 0
                        score = max(0, 100 - (mean_dist * 100))
                        score = min(100, max(0, score))

                        # Check if above threshold
                        if score >= args.threshold:
                            decision_accepted = True
                            annotated_frame = None
                            if ollama_config["enabled"] or args.save_frames:
                                annotated_frame = draw_torso_annotations(frame.copy(), results)

                            if ollama_config["enabled"]:
                                try:
                                    noul, _ = call_ollama_decision(annotated_frame, ollama_config)
                                except RuntimeError as exc:
                                    cap.release()
                                    cv2.destroyAllWindows()
                                    print(f"Error: Ollama decision failed: {exc}")
                                    sys.exit(1)

                                decision_accepted = noul >= ollama_config["threshold"]
                                status = "ACCEPT" if decision_accepted else "REJECT"
                                print(f"[{minutes}:{seconds:06.3f}] NOUL: {noul:.4f}/{ollama_config['threshold']:.4f} - {status}")

                            if decision_accepted:
                                matching_entries.append({
                                    "frame": frame_count,
                                    "time_sec": timestamp,
                                    "timestamp": f"{minutes:02d}:{seconds:06.3f}",
                                    "score": round(float(score), 1),
                                })
                                print(f"[{minutes}:{seconds:06.3f}] Score: {score:.1f}/100 - MATCH")

                                # Save matched frame with drawn landmarks
                                if args.save_frames:
                                    os.makedirs(args.output_dir, exist_ok=True)
                                    fname = f"{os.path.splitext(os.path.basename(args.video_file))[0]}_{minutes:02d}m{seconds:06.3f}s.jpg"
                                    save_path = os.path.join(args.output_dir, fname)
                                    cv2.imwrite(save_path, annotated_frame)

                frame_count += 1

            # Release video capture
            cap.release()
            cv2.destroyAllWindows()

            segments = build_segments(matching_entries, skip_frames, args.skip_sec)
            manifest = {
                "video": os.path.abspath(args.video_file),
                "reference_image": os.path.abspath(args.ref_image),
                "threshold": args.threshold,
                "skip_sec": args.skip_sec,
                "skip_frames": skip_frames,
                "fps": video_fps,
                "video_width": int(video_width),
                "video_height": int(video_height),
                "total_frames": total_frames,
                "frames_scanned": frame_count,
                "samples_compared": samples_compared,
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "matches": matching_entries,
                "segments": segments,
            }

            output_dir = os.path.dirname(manifest_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2)

            print(f"\nProcessing complete!")
            print(f"Total frames: {frame_count}")
            print(f"Matching timestamps found: {len(matching_entries)}")
            print(f"Stitchable segments: {len(segments)}")
            print(f"Results saved to: {manifest_path}")

        else:  # analyze mode
            image = cv2.imread(args.image)
            if image is None:
                print(f"Error: Could not read image {args.image}")
                sys.exit(1)

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)

            if not results.pose_landmarks:
                print("No pose detected in the image.")
            else:
                annotated_image = draw_torso_annotations(image.copy(), results)

                # Show at 80% scale
                h, w = annotated_image.shape[:2]
                small_image = cv2.resize(annotated_image, (int(w * 0.8), int(h * 0.8)), interpolation=cv2.INTER_AREA)
                cv2.imshow("Pose Landmarks (80% scale)", small_image)
                print("Displaying image at 80% scale. Press any key to close.")
                cv2.waitKey(0)
                cv2.destroyAllWindows()

                print(f"\n{'Landmark':<20} | {'X (Relative)':<15} | {'Y (Relative)':<15}")
                print("-" * 55)

                for idx in TORSO_LANDMARKS:
                    landmark = results.pose_landmarks.landmark[idx]
                    name = mp_pose.PoseLandmark(idx).name
                    print(f"{name:<20} | {landmark.x:<15.4f} | {landmark.y:<15.4f}")


if __name__ == "__main__":
    main()