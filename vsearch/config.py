"""Config handling for vsearch.

The config file (TOML) grouping:
  [index]    - index-mode defaults
  [search]   - search-mode defaults

CLI flags always override config values when given.

Uses tomllib (Python 3.11+) for reading TOML files.
"""

import os
import tomllib
from pathlib import Path

CONFIG_DIR = Path(__file__).resolve().parent.parent
CONFIG_FILE = CONFIG_DIR / "config.toml"

# Config is always loaded from the repo directory (.vsearch/config.toml)
def get_config_path() -> Path:
    """Get the path to the config file, creating directories if needed."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    return CONFIG_FILE

def default_config() -> dict:
    return {
        "index": {
            "output_dir": ".",
            "model": "openai/clip-vit-base-patch32",
            "sample_rate": 1
        },
        "search": {
            "top_k": -1,
            "raw_floor": 0.25,
            "norm_threshold": 0.85,
            "json_output": False,
            "save_frames": None,
            "video": None,
            "decision_model_validate": False,
            "ollama_endpoint": "http://localhost:11434/v1/systemone",
            "ollama_decision_model": "clef-flash",
            "decision_model_threshold": 0.5
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
            print(f"Warning: Could not parse config file {config_path}: {e}")

    return config

def save_config(config: dict) -> None:
    """Save configuration to file."""
    config_path = get_config_path()
    with open(config_path, "w") as f:
        tomllib.dump(config, f)

def get_config_value(config: dict, section: str, key: str, cli_value=None):
    """Get config value, using cli_value if provided, else config/index/search then global."""
    if cli_value is not None:
        return cli_value
    return config.get(section, {}).get(key, config.get("global", {}).get(key))

def set_config_value(config: dict, section: str, key: str, value) -> None:
    """Set config value."""
    if section not in config:
        config[section] = {}
    config[section][key] = value