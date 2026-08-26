"""
Centralized Path Utilities for Enterprise AI Training Pipelines.
Provides canonical Pathlib resolution and portable POSIX serialization.
"""

from pathlib import Path

# Canonical Project Root Resolution (GATEWAY root directory)
PROJECT_ROOT = Path(__file__).resolve().parents[3]


def resolve_path(path_input: str | Path, base_dir: Path = PROJECT_ROOT) -> Path:
    """
    Resolves any absolute or relative path securely against the project root.
    If the path is already absolute, returns it directly.
    If relative, anchors it to base_dir (defaults to PROJECT_ROOT).
    """
    p = Path(path_input)
    if p.is_absolute():
        return p
    return (base_dir / p).resolve()


def to_portable_path(path_input: str | Path, base_dir: Path = PROJECT_ROOT) -> str:
    """
    Converts an absolute or relative path to a normalized, portable POSIX relative path
    for clean JSON manifests, reports, and cross-platform compatibility.
    """
    p = Path(path_input).resolve()
    try:
        return p.relative_to(base_dir.resolve()).as_posix()
    except ValueError:
        return p.as_posix()
