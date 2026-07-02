"""
core/config_loader.py

Centralized YAML configuration loader.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from functools import lru_cache

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIGS_DIR = PROJECT_ROOT / "configs"


@lru_cache(maxsize=None)
def load_config(config_name: str) -> dict[str, Any]:
    """
    Load a YAML config file by name (without .yaml extension).
    """
    config_path = CONFIGS_DIR / f"{config_name}.yaml"

    if not config_path.exists():
        raise FileNotFoundError(
            f"Config not found: {config_path}\n"
            f"Available configs: {list(CONFIGS_DIR.glob('*.yaml'))}"
        )

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config


def get_all_configs() -> dict[str, dict[str, Any]]:
    """Load all available config files at once."""
    configs = {}
    for path in CONFIGS_DIR.glob("*.yaml"):
        name = path.stem
        configs[name] = load_config(name)
    return configs


if __name__ == "__main__":
    print("=" * 60)
    print("Config Loader Smoke Test")
    print("=" * 60)

    all_configs = get_all_configs()
    print(f"\nFound {len(all_configs)} config files:")
    for name in all_configs:
        print(f"  - {name}")

    print("\nSample: data_config")
    data_cfg = load_config("data_config")
    print(f"  start_date: {data_cfg['data_collection']['start_date']}")
    print(f"  intervals: {data_cfg['data_collection']['intervals']}")

    print("\n✅ Config loader working")