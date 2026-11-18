"""Config handling for vsearch.

The config file contains per-mode sections:
  [search]   — decision-model-only search mode defaults
  [classify] — classify (choice-question) mode defaults

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
        "search": {
            "top_k": -1,
            "json_output": False,
            "save_frames": None,
            "decision_model_floor": 0.5,
            "decision_model_sample_rate": 1,
            "ollama_host": "192.168.1.190",
            "ollama_frame_scale": 0.2,
            "ollama_decision_model": "clef-flash:9b",
            "ollama_timeout": 60,
            "ollama_state_template": "",
            "ollama_instructions_template": "",
            "ollama_search_question": "is_valid",
            "ollama_search_question_type": "noul",
            "cache_dir": "/tmp",
            "save_cache_dir": "/tmp",
        },
        "classify": {
            "top_k": -1,
            "json_output": False,
            "decision_model_floor": 0.5,
            "decision_model_sample_rate": 3,
            "ollama_host": "192.168.1.190",
            "ollama_frame_scale": 0.1,
            "ollama_decision_model": "typesafe/jev-1.14",
            "ollama_timeout": 60,
            "ollama_state_template": "sample frame from a video",
            "ollama_instructions_template": "",
            "ollama_classify_question": "team",
            "ollama_classify_instructions": "Which team should handle this scene?",
            "ollama_classify_criteria": "",
            "cache_dir": "/tmp",
            "save_cache_dir": "/tmp",
            "still_frame_threshold": 0.95,
            "same_scene_threshold": 0.80,
            "max_scene_batch_size": 4,
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
