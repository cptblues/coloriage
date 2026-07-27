"""Lanceur pratique, utilisable sans installer le paquet en mode éditable."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from coloriage_lot1.cli import main  # noqa: E402


if __name__ == "__main__":
    main()

