"""Config handling for scs.

Config file contains a flat set of keys for all defaults.
CLI flags always override config values when given.

Uses tomllib (Python 3.11+) for reading TOML files.
Loads .env file automatically if present (python-dotenv).
"""

import sys
import tomllib
from pathlib import Path
from dotenv import load_dotenv

# Load .env file if it exists (for environment variables like OPENROUTER_API_KEY)
load_dotenv()

CONFIG_DIR = Path(__file__).resolve().parent.parent
CONFIG_FILE = CONFIG_DIR / "config.toml"


def get_config_path() -> Path:
    """Get the path to the config file, creating directories if needed."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    return CONFIG_FILE


def default_config() -> dict:
    return {
        "frame_sample_rate": 1,
        "provider": "ollama",
        # Provider-agnostic frame processing settings (used by both Ollama and OpenRouter)
        "frame_scale": 0.1,
        "decision_model": "typesafe/jev-1.14",
        "state_template": "sample frame from a video",
        "frame_history": 5,
        # Ollama-specific settings
        "ollama_host": "192.168.1.190",
        "ollama_timeout": 60,
        "openrouter_api_key": "",
        "openrouter_model": "perplexity/pplx-decider-v1.3",
        "openrouter_endpoint": "https://openrouter.ai/api/alpha/decisions",
        "openrouter_timeout": 60,
        "max_images_per_request": -1,
        # Override specific models in config.toml under [model_image_limits]
        # Unlisted models fall back to built-in defaults (see scs/cli.py _BUILTIN_MODEL_LIMITS)
        "model_image_limits": {},
        "merge_threshold": 0,
        "cache": "./.cache.db",
        "still_frame_threshold": 0.95,
        "same_scene_threshold": 0.80,
        "recursive": False,
        "questions": [
            {
                "name": "activity",
                "instructions": "What activity is happening in this frame?",
                "criteria": {
                    "walking": "Person walking",
                    "running": "Person running",
                    "sitting": "Person sitting",
                },
            }
        ],
    }


def _flatten_config(user_cfg: dict) -> dict:
    """Flatten nested [ollama] and [openrouter] TOML sections into flat keys.

    TOML sections like [ollama] produce nested dicts, but the rest of the
    code expects flat keys like "ollama_host".  This helper bridges that gap.
    """
    flat = {}
    for section in ("ollama", "openrouter"):
        section_data = user_cfg.get(section)
        if isinstance(section_data, dict):
            for key, value in section_data.items():
                flat[f"{section}_{key}"] = value
    # Copy top-level keys (provider, cache, questions, model_image_limits, etc.)
    for key, value in user_cfg.items():
        if key not in ("ollama", "openrouter"):
            flat[key] = value
    return flat


def load_config() -> dict:
    """Load configuration from file or return defaults if file doesn't exist."""
    config_path = get_config_path()
    config = default_config()

    if config_path.exists():
        try:
            with open(config_path, "rb") as f:
                user_cfg = tomllib.load(f)
            config.update(_flatten_config(user_cfg))
        except Exception as e:
            print(f"Warning: Could not parse config file {config_path}: {e}", file=sys.stderr)

    return config


def get_config_value(config: dict, key: str, cli_value=None):
    """Get config value, using cli_value if provided, else config value."""
    if cli_value is not None:
        return cli_value
    return config.get(key)


def get_questions(config: dict, cli_questions_json: str = None) -> list[dict]:
    """Return the list of question definitions.

    If cli_questions_json is provided, parse it as a JSON array and use that.
    Otherwise, fall back to config["questions"].
    """
    if cli_questions_json:
        import json
        try:
            parsed = json.loads(cli_questions_json)
            if isinstance(parsed, list):
                return parsed
            print("Warning: --questions must be a JSON array; ignoring", file=sys.stderr)
        except json.JSONDecodeError as e:
            print(f"Warning: could not parse --questions JSON: {e}; ignoring", file=sys.stderr)
    return config.get("questions", [])