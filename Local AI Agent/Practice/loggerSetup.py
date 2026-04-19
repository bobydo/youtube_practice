import logging
import os
from datetime import datetime


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.DEBUG)

    os.makedirs("logs", exist_ok=True)
    filename = f"logs/agent_{datetime.now():%Y-%m-%d}.log"

    fmt = logging.Formatter(
        "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    fh = logging.FileHandler(filename, encoding="utf-8")
    fh.setFormatter(fmt)
    # Remove StreamHandler → silent terminal, logs only to file
    # ch = logging.StreamHandler()
    # ch.setFormatter(fmt)

    logger.addHandler(fh)
    # logger.addHandler(ch)
    return logger
