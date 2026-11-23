"""Composants PySide6 réutilisables (cartes KPI, badges, tableaux)."""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..theme import ACCENT, DANGER, INFO, MUTED, PRIMARY, WARNING


class KpiCard(QFrame):
    def __init__(self, label: str, value: str, sub: str = "", color: str = PRIMARY, parent=None):
        super().__init__(parent)
        self.setObjectName("Card")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(18, 14, 18, 14)
        lay.setSpacing(2)
        lbl = QLabel(label.upper())
        lbl.setObjectName("CardLabel")
        val = QLabel(value)
        val.setObjectName("CardValue")
        val.setStyleSheet(f"color: {color};")
        sub_lbl = QLabel(sub)
        sub_lbl.setStyleSheet(f"color: {MUTED}; font-size: 11px;")
        sub_lbl.setWordWrap(True)
        lay.addWidget(lbl)
        lay.addWidget(val)
        lay.addWidget(sub_lbl)


class Badge(QLabel):
    COLORS = {"SUCCESS": ACCENT, "FAILURE": DANGER, "PENDING": WARNING}

    def __init__(self, status: str, parent=None):
        super().__init__(self.text_for(status), parent)
        color = self.COLORS.get(status, INFO)
        self.setStyleSheet(
            f"color: {color}; border: 1px solid {color}; border-radius: 10px; padding: 3px 10px; font-weight: 700;"
        )

    @staticmethod
    def text_for(status: str) -> str:
        return {"SUCCESS": "SUCCÈS", "FAILURE": "ÉCHEC", "PENDING": "EN ATTENTE"}.get(status, status)


def table(headers: list[str], rows: list[list[str]], height: int = 320) -> QTableWidget:
    t = QTableWidget(len(rows), len(headers))
    t.setHorizontalHeaderLabels(headers)
    for i, row in enumerate(rows):
        for j, value in enumerate(row):
            item = QTableWidgetItem("" if value is None else str(value))
            item.setFlags(item.flags() & ~Qt.ItemIsEditable)
            t.setItem(i, j, item)
    t.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
    t.verticalHeader().setVisible(False)
    t.setMinimumHeight(height)
    t.setAlternatingRowColors(True)
    return t


def card_row(cards: list[QWidget]) -> QWidget:
    w = QWidget()
    lay = QHBoxLayout(w)
    lay.setContentsMargins(0, 0, 0, 0)
    lay.setSpacing(12)
    for c in cards:
        lay.addWidget(c)
    return w


def heading(text: str, size: int = 18) -> QLabel:
    lbl = QLabel(text)
    lbl.setStyleSheet(f"font-size: {size}px; font-weight: 700; color: #FFFFFF;")
    return lbl


def muted(text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setStyleSheet(f"color: {MUTED}; font-size: 12px;")
    lbl.setWordWrap(True)
    return lbl
