"""Make the repository root importable so that `src.meta...` and the generator resolve.

If tests/conftest.py already exists from Checkpoint 2 or 3, keep the old content and add
only the lines below.
"""
import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for _p in (_ROOT, os.path.join(_ROOT, "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
