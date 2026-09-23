"""Compatibility entry point for the independently verified saved-result audit."""
from pathlib import Path
import runpy

if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).resolve().with_name("audit_seeded_results.py")), run_name="__main__")
