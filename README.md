# Pose Search Tool

Analyze and compare human body poses in images using MediaPipe. This tool is built for finding moments in a video where a person’s torso has a pose similar to a reference image.

## What It Does

- **Analyze** — Detect torso landmarks in one image.
- **Compare** — Compare poses in two images and print a 0–100 score.
- **Video search** — Scan a video, find timestamps matching a reference pose, and write a JSON manifest.
- **Stitching helper** — Use the manifest’s `segments` list to cut and concatenate matching parts with FFmpeg.

## Recommended Workflow

```text
reference.jpg
     │
     ▼
pose_search.py video ──► pose match ──► Ollama NOUL decision ──► input.matches.json ──► FFmpeg ──► stitched.mp4
     │                         │
     │                         └─ reject below-threshold frames
     └─► console output
```

1. Choose a clear reference image with one clearly visible person.
2. Run video search with the settings from `pose_search.config.json` (or override them on the CLI).
3. MediaPipe first checks whether the frame matches the reference pose.
4. If it matches, Ollama receives the annotated frame and returns a NOUL decision score.
5. Frames accepted by Ollama are written to `.matches.json`.
6. Use FFmpeg with the manifest’s `segments` to create the final stitched video.

## Requirements

| Requirement | Recommended version | Purpose |
|---|---:|---|
| Python | 3.10–3.12 | MediaPipe compatibility |
| OpenCV (`opencv-contrib-python`) | `>=4.8.0,<5.0.0` | Image/video I/O and drawing |
| NumPy | `>=1.24.0` | Landmark math |
| MediaPipe | `==0.10.21` | Pose detection |
| FFmpeg | Recent stable build | Cutting and concatenating video |
| Local Ollama | Installed and running if `ollama.enabled` is true | NOUL frame decision |

> Python 3.13+ is not recommended because current MediaPipe releases may be incompatible with it.

### Important: current `requirements.txt`

The repository’s `requirements.txt` currently lists OpenCV and NumPy, but **not MediaPipe**. Install MediaPipe explicitly:

```powershell
pip install -r requirements.txt
pip install mediapipe==0.10.21
```

FFmpeg is also not installed by pip. Install it separately and verify it is available:

```powershell
ffmpeg -version
```

On Windows, FFmpeg can be installed with your package manager or from [ffmpeg.org](https://ffmpeg.org/download.html).

## Install

From the project directory:

```powershell
# Optional but recommended: create an isolated environment
py -3.11 -m venv venv
.\venv\Scripts\Activate.ps1

# Install Python dependencies
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install mediapipe==0.10.21
```

Verify the command works:

```powershell
python pose_search.py --help
```

## Quick Start

### 1. Analyze one image

```powershell
python pose_search.py analyze reference.jpg
```

### 2. Compare two images

```powershell
python pose_search.py compare pose1.jpg pose2.jpg
```

### 3. Search a video

```powershell
python pose_search.py video reference.jpg input.mp4
```

Uses `pose_search.config.json`. By default, this writes `input.matches.json` in the current directory.

### 4. Stitch the matching video parts

Use the FFmpeg workflow in [Stitching with FFmpeg](#stitching-with-ffmpeg).

## Video Configuration File

`video` mode requires a fixed config file beside `pose_search.py`:

```text
pose_search.config.json
```

Command-line options override values from this file. Options omitted from the command line use the config defaults.

Example:

```json
{
  "video": {
    "threshold": 90.0,
    "skip_sec": 3.0,
    "scale": 0.2,
    "output": null,
    "confidence": 0.9,
    "complexity": 0,
    "threads": 1,
    "save_frames": true,
    "output_dir": "."
  },
  "ollama": {
    "enabled": true,
    "endpoint": "http://localhost:11434/v1/systemone",
    "model": "d3:4b",
    "state": "...",
    "question": {
      "type": "noul",
      "instructions": "...",
      "criteria": {
        "false": "",
        "true": ""
      }
    },
    "threshold": 0.8,
    "timeout_sec": 60.0
  },
  "analyze": {
    "confidence": 0.5,
    "complexity": 1
  },
  "compare": {
    "confidence": 0.5,
    "complexity": 1,
    "mirror": false
  }
}

### `video` settings

| Key | Meaning |
|---|---|
| `threshold` | MediaPipe pose score accepted before the Ollama decision |
| `skip_sec` | Seconds between sampled frames |
| `scale` | Scale applied only to detection frames |
| `output` | Manifest path; `null` means `<video-name>.matches.json` |
| `confidence` | MediaPipe minimum detection confidence |
| `complexity` | MediaPipe model complexity: 0, 1, or 2 |
| `threads` | Accepted for compatibility; currently unused |
| `save_frames` | Save annotated matched frames |
| `output_dir` | Directory for saved frames |

### `compare` settings

| Key | Meaning |
|---|---|
| `confidence` | MediaPipe minimum detection confidence (0.0–1.0) |
| `complexity` | MediaPipe model complexity: 0, 1, or 2 |
| `mirror` | Flip image2 horizontally before comparison |

### `ollama` settings

| Key | Meaning |
|---|---|
| `enabled` | Call Ollama after a frame passes the pose threshold |
| `endpoint` | HTTP POST endpoint, for example `http://localhost:11434/v1/systemone` |
| `model` | Model name sent in the request |
| `state` | Request state/context string |
| `question` | Single `noul` question object copied into the request as `{"question": {...}}` |
| `threshold` | Raw NOUL value required to accept the frame |
| `timeout_sec` | HTTP request timeout |

The frame sent to Ollama is the annotated image: the frame with green torso lines and landmark points. The `state` string should describe that image format.

NOUL comparison:

```text
noul >= threshold
```

For example, `threshold: 0.8` accepts `0.91` and rejects `0.39`.

### Error behavior

Video mode fails fast when:

- `pose_search.config.json` is missing;
- JSON is invalid;
- required settings are missing or have the wrong type;
- the `question` is missing or not an object;
- the `question` is not a `noul` question;
- Ollama cannot be reached;
- Ollama returns invalid JSON or no numeric `noul` value.

`analyze` and `compare` read their sections (`analyze`, `compare`) from the same config file as defaults — when the file is missing or invalid, they fall back to built-in defaults rather than failing. CLI options override the config.

## Commands

### Analyze a Single Image

```powershell
python pose_search.py analyze "path\to\image.jpg"
```

**Options:**

| Flag | Config default | Description |
|------|---------|-------------|
| `--confidence` | `0.5` | Minimum detection confidence (0.0–1.0) |
| `--complexity` | `1` | Model complexity: `0` fast, `1` balanced, `2` accurate |

**Output:**

- Console: relative coordinates `(x, y)` for the four torso landmarks
- Window: annotated image at 80% scale

### Compare Two Images

```powershell
python pose_search.py compare "path\to\image1.jpg" "path\to\image2.jpg"
```

**Options:**

| Flag | Config default | Description |
|------|---------|-------------|
| `--confidence` | `0.5` | Minimum detection confidence (0.0–1.0) |
| `--complexity` | `1` | Model complexity: `0` fast, `1` balanced, `2` accurate |
| `--mirror` / `--no-mirror` | `false` | Flip image2 horizontally before comparison |

**Output:**

- Console: per-landmark differences, score, and match classification
- Window: overlay blend of both images at 80% scale

### Video Search

```powershell
python pose_search.py video reference.jpg input.mp4
```

By default, settings are read from `pose_search.config.json`. Command-line options override the config for that run.

**Default output:** `<video-name>.matches.json` in the current directory, unless `video.output` is set in the config or `--output` is passed.

**Options:**

| Flag | Config default | Description |
|------|---------|-------------|
| `--threshold` | `90.0` | Minimum pose score (0–100) required before the Ollama decision |
| `--skip-sec` | `3.0` | Seconds between compared frames |
| `--scale` | `0.2` | Detection-frame scale (0.1–1.0); lower values are faster but may reduce accuracy |
| `--output` | `null` | Custom JSON manifest path |
| `--confidence` | `0.9` | Minimum detection confidence (0.0–1.0) |
| `--complexity` | `0` | Model complexity: `0` fast, `1` balanced, `2` accurate |
| `--save-frames` / `--no-save-frames` | `true` | Save or do not save annotated matched frames |
| `--output-dir` | `.` | Directory for saved matched frames |
| `--threads` | `1` | Accepted for compatibility; currently not used |

**Output:**

- Console: matching timestamps, match count, segment count, and accept/reject decisions
- JSON manifest: machine-readable matches and stitchable segments

## Choosing Settings

### `--threshold`

| Threshold | Result |
|---|---|
| `95` | Strict; fewer but stronger matches |
| `90` | Balanced; used in the current workflow |
| `80` | Broader; may include weak or incorrect matches |
| `<80` | Usually too loose for automatic stitching |

### `--skip-sec`

Controls how often frames are compared.

| Value | Effect |
|---:|---|
| `1.0` | More precise, slower |
| `3.0` | Good balance; used in the current workflow |
| `5.0+` | Faster, but short matching moments may be missed |

Matches are sampled at every Nth frame, where N is derived from `skip-sec × video FPS`.

### `--scale`

Resizes frames only for detection. It does **not** alter the final video.

- `1.0` — most accurate, slowest
- `0.75` — moderately faster
- `0.5` — much faster, but pose detection may become less reliable

### Reference image best practices

Use a reference image where:

- one person is clearly the subject;
- shoulders and hips are visible and unobstructed;
- pose and framing are similar to the target footage;
- the subject is not too small in the frame;
- the image is not heavily cropped or rotated;
- the image has not been mirrored when exact orientation matters.

The detector finds the most prominent person. Background people are ignored only if they are less prominent.

## JSON Manifest

The video mode writes JSON because it is safer for automation than parsing a human-readable text file.

Example:

```json
{
  "video": "F:\\path\\to\\input.mp4",
  "reference_image": "F:\\path\\to\\reference.jpg",
  "threshold": 90.0,
  "skip_sec": 3.0,
  "skip_frames": 179,
  "fps": 59.94,
  "video_width": 1920,
  "video_height": 1080,
  "total_frames": 220522,
  "frames_scanned": 220522,
  "samples_compared": 1232,
  "created_at": "2026-11-04T00:02:53",
  "matches": [
    {
      "frame": 5728,
      "time_sec": 95.56213333333334,
      "timestamp": "01:35.562",
      "score": 92.7
    }
  ],
  "segments": [
    {
      "start_sec": 95.56213333333334,
      "end_sec": 101.54845,
      "duration_sec": 5.986316666666667,
      "match_frames": [5728, 5907],
      "match_scores": [92.7, 92.3]
    }
  ]
}
```

### Manifest fields

| Field | Meaning |
|---|---|
| `video` | Absolute source video path |
| `reference_image` | Absolute reference image path |
| `threshold` | Minimum accepted score |
| `skip_sec` | Requested seconds between samples |
| `skip_frames` | Actual number of frames between samples |
| `fps` | Video frame rate reported by OpenCV |
| `total_frames` | Container-reported total frame count |
| `frames_scanned` | Number of frames read from the video |
| `samples_compared` | Number of frames actually tested |
| `created_at` | Local manifest creation timestamp |
| `matches` | All samples meeting the threshold |
| `segments` | Pre-grouped stitchable intervals |

### Segment grouping

Matches are grouped using frame indices, not string timestamps.

- If the next match is within one sampling step of the previous match, it is merged into the current segment.
- If there is a larger gap, a new segment starts.
- An isolated segment lasts `skip_sec`.
- A sequential segment ends at `last_match.time_sec + skip_sec`.

This means the stitching step does not need to reimplement grouping logic.

## Stitching with FFmpeg

The manifest contains everything needed to cut the matching video parts.

### Fast stitching: stream copy

Fast, preserves original video quality, but cuts snap to keyframes.

```powershell
$manifestPath = '.\input.matches.json'
$manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
$src = $manifest.video
$stem = [IO.Path]::GetFileNameWithoutExtension($src)
$out = Join-Path (Get-Location) "${stem}_stitched.mp4"
$tmp = Join-Path $env:TEMP "pose_stitch_$stem"
New-Item -ItemType Directory -Force -Path $tmp | Out-Null

$names = @()
for ($i = 0; $i -lt $manifest.segments.Count; $i++) {
    $seg = $manifest.segments[$i]
    $name = 'seg_{0:D2}.mp4' -f ($i + 1)
    $target = Join-Path $tmp $name
    $start = $seg.start_sec.ToString('0.###', [Globalization.CultureInfo]::InvariantCulture)
    $duration = $seg.duration_sec.ToString('0.###', [Globalization.CultureInfo]::InvariantCulture)

    ffmpeg -y -fflags +genpts -ss $start -i $src -t $duration -map 0 -c copy -avoid_negative_ts make_zero -movflags +faststart $target
    if ($LASTEXITCODE -ne 0) { throw "FFmpeg failed while creating $name" }
    $names += $name
}

$concatList = Join-Path $tmp 'concat.txt'
$names | ForEach-Object { "file '$_'" } | Set-Content -LiteralPath $concatList -Encoding ascii

Push-Location $tmp
ffmpeg -y -fflags +genpts -f concat -safe 0 -i $concatList -map 0 -c copy -movflags +faststart $out
$code = $LASTEXITCODE
Pop-Location
if ($code -ne 0) { throw 'FFmpeg concat failed' }

$out
```

Expected behavior:

- creates temporary clips in your system temp directory;
- overwrites an existing `seg_*.mp4` in that temp directory;
- creates `<video-name>_stitched.mp4` in the current directory;
- does **not** re-encode the video.

### Precise stitching: re-encode

Use this when cuts must start and end close to the requested millisecond timestamps.

Replace this line in the script:

```powershell
ffmpeg -y -fflags +genpts -ss $start -i $src -t $duration -map 0 -c copy -avoid_negative_ts make_zero -movflags +faststart $target
```

with:

```powershell
ffmpeg -y -fflags +genpts -ss $start -i $src -t $duration -map 0 -c:v libx264 -crf 18 -preset medium -c:a aac -b:a 192k $target
```

Keep the concat step using `-c copy`. This is slower than stream copy, but it avoids including extra video from the previous keyframe.

### Stream copy vs re-encoding

| Method | Speed | Quality | Cut accuracy |
|---|---|---|---|
| `-c copy` | Very fast | Original | Snaps to keyframes |
| `libx264` re-encode | Slower | Re-encoded | Frame-accurate |

For review or personal use, stream copy is usually good enough. For editing or publishing, prefer re-encoding.

## How Scoring Works

The current implementation uses **direct relative coordinates**:

1. MediaPipe Pose estimates body landmarks.
2. The tool extracts four torso landmarks: left/right shoulder and left/right hip.
3. Landmark coordinates are compared as relative image coordinates.
4. Euclidean distance is computed for the four landmark pairs.
5. The mean distance is converted into a score: `100 - (mean_distance × 100)`.
6. The score is clamped to `0–100`.

This is intentionally simple. It works best when the reference image and target video have similar framing, scale, camera angle, and subject position.

> The score is **not** a normalized body-pose embedding. Moving the same pose to a different location or scale in the frame can lower the score.

## Landmarks Used

| Index | MediaPipe name | Role |
|---:|---|---|
| 11 | `LEFT_SHOULDER` | Upper torso anchor |
| 12 | `RIGHT_SHOULDER` | Upper torso anchor |
| 23 | `LEFT_HIP` | Lower torso anchor |
| 24 | `RIGHT_HIP` | Lower torso anchor |

## Score Interpretation

| Score | Result |
|---:|---|
| 95–100 | Excellent match |
| 80–94 | Good match |
| 60–79 | Moderate match |
| <60 | Low match |

For video stitching, a threshold around `90` is a practical starting point.

## Analyze and Compare Display

The image commands use OpenCV windows. Press any key to close them.

If no window appears:

- check that you are running in an interactive desktop session;
- verify `opencv-contrib-python` is installed correctly;
- run from a local terminal rather than a headless environment.

## Troubleshooting

### `ModuleNotFoundError: No module named 'mediapipe'`

Install it explicitly:

```powershell
pip install mediapipe==0.10.21
```

### `'mediapipe.solutions' was not found`

You are probably using an incompatible Python version. Use Python 3.10, 3.11, or 3.12.

### No pose detected in the reference image

- choose a larger, clearer image;
- ensure shoulders and hips are visible;
- lower `--confidence`;
- try `--complexity 2`;
- avoid heavily occluded or blurry images.

### Too few matches

- lower `--threshold` from `90` to `85`;
- use a better reference frame;
- lower `--skip-sec`;
- try `--complexity 2`;
- confirm the video is not mirrored relative to the reference.

### Too many false positives

- raise `--threshold` to `92–95`;
- use a more specific reference image;
- check whether the reference pose is too generic;
- inspect saved frames with `--save-frames`.

### Cuts include extra seconds of footage

This is expected with `-c copy`. FFmpeg cannot remove frames before a keyframe when using stream copy. Re-encode for precise cuts.

### Final duration differs from manifest duration

The manifest duration is the requested edit duration. A stream-copied output may be longer because each segment starts at a keyframe. This is normal.

### Manifest has fewer matches than expected

Check:

- correct video file path;
- correct reference image;
- threshold value;
- pose visibility in the video;
- whether the video is horizontally mirrored.

### Existing `matches.txt` files

`matches.txt` was the old text output. New video runs write `<video-name>.matches.json`; existing `.matches.json` files are not retroactively created from old text reports.

## Project Layout

```text
pose_search.py                 Main CLI application
requirements.txt               OpenCV and NumPy dependencies
README.md                      This guide
matches.txt                    Legacy output from older versions
*.matches.json                 Machine-readable video-search manifests
*.jpg / *.png                  Example or reference images
.venv / venv                   Optional local virtual environments
```

## Current Limitations

- Detects only one person: the most prominent person in the frame.
- Uses torso landmarks only; arms, legs, and head are ignored.
- Uses direct 2D coordinates, not a normalized pose representation.
- Framing and scale have a large effect on scores.
- Front/back ambiguity exists in 2D pose estimation.
- CPU-only inference; long videos can take a long time.
- FFmpeg stream copy trades cut accuracy for speed and quality preservation.
- Python 3.13+ may not work with the pinned MediaPipe version.

## Suggested Future Improvements

- Add an explicit `stitch` subcommand to run the FFmpeg workflow directly.
- Use midpoint centering and shoulder-width normalization for scale-invariant scoring.
- Add pose orientation detection for better front/back handling.
- Add progress display for long video scans.
- Add optional audio-aware segment padding for smoother edits.
