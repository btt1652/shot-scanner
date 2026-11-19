"""Config handling for vsearch.

The config file contains a [classify] section for choice-question
classification defaults.

CLI flags always override config values when given.

Uses tomllib (Python 3.11+) for reading TOML files.
"""

import sys
import tomllib
from pathlib import Path

CONFIG_DIR = Path(__file__).resolve().parent.parent
CONFIG_FILE = CONFIG_DIR / "config.toml"


def get_config_path() -> Path:
    """Get the path to the config file, creating directories if needed."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    return CONFIG_FILE


def default_config() -> dict:
    return {
        "classify": {
            "decision_model_sample_rate": 3,
            "ollama_host": "192.168.1.190",
            "ollama_frame_scale": 0.1,
            "ollama_decision_model": "typesafe/jev-1.14",
            "ollama_timeout": 60,
            "ollama_state_template": "sample frame from a video",
            "cache_file": "./.cache.db",
            "still_frame_threshold": 0.95,
            "same_scene_threshold": 0.80,
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
        },
    }


def load_config() -> dict:
    """Load configuration from file or return defaults if file doesn't exist."""
    config_path = get_config_path()
    config = default_config()

    if config_path.exists():
        try:
            with open(config_path, "rb") as f:
                user_cfg = tomllib.load(f)
            for section, values in user_cfg.items():
                if isinstance(values, dict):
                    merged = config.get(section, {})
                    merged.update(values)
                    config[section] = merged
        except Exception as e:
            print(f"Warning: Could not parse config file {config_path}: {e}", file=sys.stderr)

    return config


def get_config_value(config: dict, section: str, key: str, cli_value=None):
    """Get config value, using cli_value if provided, else config/search then global."""
    if cli_value is not None:
        return cli_value
    return config.get(section, {}).get(key, config.get("global", {}).get(key))


def get_questions(config: dict, cli_questions_json: str = None) -> list[dict]:
    """Return the list of question definitions.

    If cli_questions_json is provided, parse it as a JSON array and use that.
    Otherwise, fall back to config["classify"]["questions"].
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
    return config.get("classify", {}).get("questions", [])
