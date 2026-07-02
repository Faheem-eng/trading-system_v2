"""
core/logger.py

Structured logging setup for the entire project.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

_LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Get a configured logger for a module.
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(level)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(
        logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)
    )
    logger.addHandler(console_handler)

    file_handler = logging.FileHandler(
        LOG_DIR / "trading_system.log", encoding="utf-8"
    )
    file_handler.setFormatter(
        logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)
    )
    logger.addHandler(file_handler)

    return logger


if __name__ == "__main__":
    logger = get_logger(__name__)
    logger.info("Logger smoke test — INFO level")
    logger.warning("Logger smoke test — WARNING level")
    logger.error("Logger smoke test — ERROR level")
    print("\n✅ Check logs/trading_system.log for file output")