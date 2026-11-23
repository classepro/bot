"""Thème visuel partagé (couleurs, CSS Streamlit, styles PySide6)."""
from __future__ import annotations

PRIMARY = "#6C5CE7"
ACCENT = "#00B894"
WARNING = "#FDCB6E"
DANGER = "#E17055"
INFO = "#0984E3"
MUTED = "#8E8EA0"
BG = "#0E1117"
CARD = "#161B22"

STREAMLIT_CSS = f"""
<style>
:root {{
  --primary: {PRIMARY}; --accent: {ACCENT}; --warning: {WARNING};
  --danger: {DANGER}; --muted: {MUTED}; --card: {CARD};
}}
.block-container {{ padding-top: 1.2rem; max-width: 1500px; }}
h1, h2, h3 {{ letter-spacing: -0.02em; }}
.cp-hero {{
  background: linear-gradient(120deg, rgba(108,92,231,.25), rgba(0,184,148,.12));
  border: 1px solid rgba(108,92,231,.35); border-radius: 18px; padding: 22px 26px; margin-bottom: 18px;
}}
.cp-hero h1 {{ margin: 0; font-size: 1.9rem; }}
.cp-hero p {{ margin: .35rem 0 0; color: #C9C9D6; }}
.cp-card {{
  background: {CARD}; border: 1px solid rgba(255,255,255,.08); border-radius: 16px;
  padding: 18px 20px; height: 100%;
}}
.cp-card .cp-label {{ text-transform: uppercase; letter-spacing: .08em; font-size: .72rem; color: {MUTED}; }}
.cp-card .cp-value {{ font-size: 1.85rem; font-weight: 700; line-height: 1.15; }}
.cp-card .cp-sub {{ font-size: .8rem; color: {MUTED}; }}
.cp-pill {{ display: inline-block; padding: 3px 10px; border-radius: 999px; font-size: .75rem; font-weight: 600; }}
.cp-success {{ background: rgba(0,184,148,.16); color: {ACCENT}; border: 1px solid rgba(0,184,148,.45); }}
.cp-failure {{ background: rgba(225,112,85,.16); color: {DANGER}; border: 1px solid rgba(225,112,85,.45); }}
.cp-pending {{ background: rgba(253,203,110,.14); color: {WARNING}; border: 1px solid rgba(253,203,110,.45); }}
.cp-info {{ background: rgba(9,132,227,.14); color: {INFO}; border: 1px solid rgba(9,132,227,.4); }}
.cp-warn {{ background: rgba(253,203,110,.1); border-left: 4px solid {WARNING}; padding: 12px 16px; border-radius: 10px; }}
.cp-good {{ background: rgba(0,184,148,.08); border-left: 4px solid {ACCENT}; padding: 12px 16px; border-radius: 10px; }}
.cp-bad {{ background: rgba(225,112,85,.08); border-left: 4px solid {DANGER}; padding: 12px 16px; border-radius: 10px; }}
.cp-mono {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: .78rem; color: {MUTED}; }}
section[data-testid="stSidebar"] {{ border-right: 1px solid rgba(255,255,255,.07); }}
</style>
"""


def pyside_stylesheet() -> str:
    return f"""
    QWidget {{ background-color: {BG}; color: #E6E6EF; font-family: 'Segoe UI', 'Inter', sans-serif; font-size: 13px; }}
    QMainWindow, QDialog {{ background-color: {BG}; }}
    #Sidebar {{ background-color: #0B0E14; border-right: 1px solid #23283A; }}
    #Brand {{ font-size: 18px; font-weight: 700; color: {PRIMARY}; padding: 14px; }}
    QPushButton#NavButton {{ text-align: left; padding: 10px 16px; border: none; border-radius: 8px; color: #B9B9C9; }}
    QPushButton#NavButton:hover {{ background-color: rgba(108,92,231,.18); color: #FFFFFF; }}
    QPushButton#NavButton:checked {{ background-color: {PRIMARY}; color: white; font-weight: 600; }}
    QPushButton {{ background-color: {PRIMARY}; border: none; border-radius: 8px; padding: 8px 16px; color: white; font-weight: 600; }}
    QPushButton:hover {{ background-color: #7D6EEB; }}
    QPushButton#Secondary {{ background-color: #23283A; }}
    QPushButton#Secondary:hover {{ background-color: #2E3450; }}
    QFrame#Card {{ background-color: {CARD}; border: 1px solid #23283A; border-radius: 14px; }}
    QLabel#CardLabel {{ color: {MUTED}; font-size: 11px; text-transform: uppercase; letter-spacing: 1px; }}
    QLabel#CardValue {{ font-size: 26px; font-weight: 700; }}
    QLabel#BadgeSuccess {{ color: {ACCENT}; font-weight: 700; }}
    QLabel#BadgeFailure {{ color: {DANGER}; font-weight: 700; }}
    QTableWidget {{ background-color: {CARD}; border: 1px solid #23283A; border-radius: 10px; gridline-color: #23283A; }}
    QHeaderView::section {{ background-color: #1B2130; color: #C9C9D6; border: none; padding: 6px; }}
    QComboBox, QLineEdit, QSpinBox, QDoubleSpinBox {{ background-color: #1B2130; border: 1px solid #2E3450; border-radius: 8px; padding: 6px; }}
    QProgressBar {{ background-color: #1B2130; border-radius: 6px; text-align: center; }}
    QProgressBar::chunk {{ background-color: {PRIMARY}; border-radius: 6px; }}
    """
