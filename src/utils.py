"""
utils.py
--------
Utility functions and configuration helpers for the E-Waste Project.
Provides reusable functions such as project path management and directory validation.
"""

from pathlib import Path
import logging


def get_project_root() -> Path:
    """
    Returns the absolute Path object representing the project root directory.
    This ensures file paths remain reliable regardless of where scripts are run from.
    """
    # Assuming this file is located at <project_root>/src/utils.py
    return Path(__file__).resolve().parent.parent


def ensure_project_directories() -> None:
    """
    Ensures that essential project directories exist.
    Creates them if they are missing.
    """
    root = get_project_root()
    required_dirs = ["Dataset", "src", "graphs", "results", "reports"]
    
    for dir_name in required_dirs:
        dir_path = root / dir_name
        dir_path.mkdir(parents=True, exist_ok=True)


def setup_logger(name: str = "EWasteProject") -> logging.Logger:
    """
    Sets up and returns a standard formatted logger for the project.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    return logger
