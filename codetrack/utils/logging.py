"""Console + file logging setup, one log per run under outputs/<exp>/."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional, Union


def get_logger(name: str = "codetrack",
               log_file: Optional[Union[str, Path]] = None) -> logging.Logger:
    """Return a stdout logger, optionally also writing to ``log_file``."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("[%(asctime)s] %(levelname)s %(message)s", "%H:%M:%S")

    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(fmt)
    logger.addHandler(stream)

    if log_file:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(str(path), encoding="utf-8")
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)
    return logger
