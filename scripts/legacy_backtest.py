#!/usr/bin/env python
"""Évalue ROBOT V4.5 (legacy/) et enregistre le rapport robotv45.json."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from crashpredict.legacy.run import main

if __name__ == "__main__":
    raise SystemExit(main())
