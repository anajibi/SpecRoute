from __future__ import annotations

import logging


def setup_logging() -> logging.Logger:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    return logging.getLogger(__name__)

