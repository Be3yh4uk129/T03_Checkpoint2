"""Make the packages under src/ importable when pytest is run from the repository root.

If this file already exists in the repository from Checkpoint 2, keep the old
content and add only the three lines that extend sys.path.
"""
import os
import sys

_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
