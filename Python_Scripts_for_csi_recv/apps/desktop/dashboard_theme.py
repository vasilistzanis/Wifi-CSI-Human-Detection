from __future__ import annotations

BG = "#0d1117"
PANEL = "#161b22"
BORDER = "#30363d"
DARK2 = "#21262d"
TEXT = "#c9d1d9"
DIM = "#8b949e"
BLUE = "#58a6ff"
GREEN = "#3fb950"
YELLOW = "#d29922"
RED = "#f85149"
ORANGE = "#f0883e"
PURPLE = "#a371f7"

CLASS_COLORS = {
    "empty": BLUE,
    "no_activity": GREEN,
    "walk_activity": ORANGE,
    "idle": GREEN,
    "walk": ORANGE,
    "sit": YELLOW,
    "fall": RED,
    "stand": PURPLE,
    "run": "#ff7b72",
    "calm": GREEN,
    "low": YELLOW,
    "medium": ORANGE,
    "high": RED,
}

DISPLAY_NAMES = {
    "walk": "walk/activity",
    "walk_activity": "walk/activity",
    "idle": "idle/no activity",
    "no_activity": "idle/no activity",
}

MODEL_NAMES = {
    "rf": "Random Forest",
    "svm": "SVM (RBF)",
    "et": "Extra Trees",
    "knn": "K-NN",
    "lr": "Logistic Reg.",
    "gb": "Gradient Boost",
    "mlp": "MLP Net",
    "nb": "Naive Bayes",
}

MODEL_DESC = {
    "rf": "Robust ensemble of decision trees. Good general accuracy.",
    "svm": "Support Vector Machine with RBF kernel. High CSI accuracy.",
    "et": "Extra Trees — faster training, similar accuracy to RF.",
    "knn": "K-Nearest Neighbors. Simple, interpretable, low overhead.",
    "lr": "Logistic Regression. Fastest inference, lower complexity.",
    "gb": "Gradient Boosting. High accuracy, slower inference.",
    "mlp": "Neural network (MLP). Flexible non-linear feature learning.",
    "nb": "Naive Bayes. Fastest, best for real-time constrained use.",
}


def class_color(label: str) -> str:
    return CLASS_COLORS.get(label.lower(), BLUE)


def display_name(label: str) -> str:
    return DISPLAY_NAMES.get(label.lower(), label)


STYLESHEET = f"""
QMainWindow, QWidget {{
    background:{BG}; color:{TEXT};
    font-family:'Segoe UI','Arial',sans-serif;
}}
QLabel  {{ color:{TEXT}; background:transparent; }}
QWidget#sidebar {{ border-right:1px solid {BORDER}; }}
QTableWidget {{
    background:{PANEL}; border:1px solid {BORDER}; border-radius:6px;
    gridline-color:{BORDER}; color:{TEXT}; font-size:11px;
}}
QTableWidget::item {{ padding:2px 4px; }}
QTableWidget::item:selected {{ background:{DARK2}; }}
QHeaderView::section {{
    background:{DARK2}; color:{DIM}; border:none;
    border-bottom:1px solid {BORDER}; padding:4px 8px;
    font-size:9px; font-weight:bold; letter-spacing:0.06em;
}}
QScrollBar:vertical {{ background:{BG}; width:5px; border-radius:2px; }}
QScrollBar::handle:vertical {{ background:{BORDER}; border-radius:2px; min-height:16px; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height:0px; }}
QProgressBar {{ background:{DARK2}; border:none; border-radius:3px; }}
QProgressBar::chunk {{ border-radius:3px; }}
"""

NAV_ACTIVE = (
    f"QPushButton{{background:{DARK2}; color:{BLUE}; font-weight:bold;"
    f" border-left:3px solid {BLUE}; padding-left:13px;"
    f" border-radius:0; font-size:13px; text-align:left;}}"
)
NAV_INACTIVE = (
    f"QPushButton{{background:transparent; color:{DIM};"
    f" border-left:3px solid transparent; padding-left:13px;"
    f" border-radius:0; font-size:13px; text-align:left;}}"
    f"QPushButton:hover{{background:{DARK2}; color:{TEXT};}}"
)

_BG = BG
_PANEL = PANEL
_BORDER = BORDER
_DARK2 = DARK2
_TEXT = TEXT
_DIM = DIM
_BLUE = BLUE
_GREEN = GREEN
_YELLOW = YELLOW
_RED = RED
_ORANGE = ORANGE
_MODEL_NAMES = MODEL_NAMES
_MODEL_DESC = MODEL_DESC
_QSS = STYLESHEET
_NAV_ON = NAV_ACTIVE
_NAV_OFF = NAV_INACTIVE

__all__ = [
    "_BG",
    "_BORDER",
    "_BLUE",
    "_DARK2",
    "_DIM",
    "_GREEN",
    "_MODEL_DESC",
    "_MODEL_NAMES",
    "_NAV_OFF",
    "_NAV_ON",
    "_ORANGE",
    "_PANEL",
    "_QSS",
    "_RED",
    "_TEXT",
    "_YELLOW",
    "_cc",
    "_disp",
]


def _cc(label: str) -> str:
    return class_color(label)


def _disp(label: str) -> str:
    return display_name(label)
