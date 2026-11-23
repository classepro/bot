"""Pages Streamlit (une fonction `render()` par module)."""
from __future__ import annotations

from . import (
    page_backtesting,
    page_dashboard,
    page_data,
    page_history,
    page_prediction,
    page_report,
    page_training,
)

__all__ = [
    "page_dashboard",
    "page_prediction",
    "page_history",
    "page_training",
    "page_backtesting",
    "page_data",
    "page_report",
]
