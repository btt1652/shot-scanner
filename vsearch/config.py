"""Config handling for vsearch.

The config file (TOML) contains a single [search] section for decision-model-only search defaults.

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
            "video": None,
            "decision_model_floor": 0.5,
            "decision_model_sample_rate": 1,
            "ollama_host": "192.168.1.190",
            "ollama_frame_scale": 0.2,
            "ollama_decision_model": ["d3:9b", "d3:4b"],
            "ollama_workers": 1,
            "ollama_state_template": "",
            "ollama_instructions_template": "",
            "scene_change": False,
            "scene_threshold": 0.4,
            "sample_merge": False,
            "merge_gap": None,
        }
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
