"""Structured, persisted logging - one place every entrypoint configures
from, instead of ad hoc print() calls with no audit trail.

Design: this is the operational/audit log (run lifecycle, config summary,
warnings, errors) - it answers "what happened, when, and why" after the
fact, which is exactly what report Sec 4.4 (Production Monitoring)
describes wanting for a real deployment, exercised here at the CLI-script
level. It is deliberately NOT used for the human-facing tabular reports
(print_comparison, print_breakdown in evaluate.py/error_analysis.py) -
those are the command's actual result/output, analogous to `git status`'s
output vs. `git`'s internal trace, and reformatting them as timestamped
log lines would make them harder to read for zero benefit. Both the
tabular output AND the log file exist; they answer different questions.
"""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

DEFAULT_LOG_DIR = Path("outputs/logs")
LOG_FORMAT = "%(asctime)s %(levelname)-8s %(name)s: %(message)s"


def setup_logging(
    name: str,
    log_dir: Path = DEFAULT_LOG_DIR,
    level: int = logging.INFO,
    max_bytes: int = 5 * 1024 * 1024,
    backup_count: int = 3,
) -> logging.Logger:
    """Configures and returns a logger that writes to both the console and
    a rotating file (`<log_dir>/<name>.log`). Rotation (max_bytes,
    backup_count) is the log-file analog of the checkpoint backup rotation
    in train.py - bounded disk usage instead of one unboundedly growing
    file, without ever silently losing the most recent history."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()  # idempotent: re-calling setup_logging (e.g. in tests) doesn't stack handlers

    formatter = logging.Formatter(LOG_FORMAT)

    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    log_dir.mkdir(parents=True, exist_ok=True)
    file_handler = RotatingFileHandler(
        log_dir / f"{name}.log", maxBytes=max_bytes, backupCount=backup_count
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.propagate = False  # don't also emit through the root logger's default handler
    return logger
