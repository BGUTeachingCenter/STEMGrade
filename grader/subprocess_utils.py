# grader/subprocess_utils.py
from __future__ import annotations

import subprocess
from pathlib import Path


def run(cmd: list[str], cwd: Path | None = None) -> None:
    """Run a command; if it fails, print stdout/stderr and raise."""
    try:
        subprocess.run(
            cmd,
            check=True,
            cwd=cwd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except subprocess.CalledProcessError as e:
        print("\n--- COMMAND FAILED ---")
        print("CMD:", " ".join(cmd))
        print("\nSTDOUT:\n", e.stdout)
        print("\nSTDERR:\n", e.stderr)
        raise
