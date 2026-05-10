#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shared visual theme for all thesis plotting scripts.

Import what you need:
    from plot_styles import STYLE, PALETTE, CLASS_COLORS, PCA_COLORS, MODEL_PALETTE, _apply_style
"""

import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Canonical color palette — professional light theme for thesis/paper printing
# Both full-name keys ("accent1") and short aliases ("a1") are supported.
# ---------------------------------------------------------------------------
STYLE: dict[str, str] = {
    "bg":      "#ffffff",
    "panel":   "#fafafa",
    "text":    "#1a1a1a",
    "grid":    "#e0e0e0",
    "accent1": "#2563eb",  "a1": "#2563eb",  # blue
    "accent2": "#f59e0b",  "a2": "#f59e0b",  # amber
    "accent3": "#10b981",  "a3": "#10b981",  # green
    "accent4": "#ef4444",  "a4": "#ef4444",  # red
    "accent5": "#8b5cf6",  "a5": "#8b5cf6",  # purple
    "accent6": "#06b6d4",  "a6": "#06b6d4",  # cyan
}

# Six-color ordered palette (matches accent1 → accent6)
PALETTE: list[str] = [
    STYLE["accent1"], STYLE["accent2"], STYLE["accent3"],
    STYLE["accent4"], STYLE["accent5"], STYLE["accent6"],
]

# Per-class color assignment used in scatter / bar charts
CLASS_COLORS: list[str] = [
    STYLE["accent1"], STYLE["accent3"], STYLE["accent4"],
    STYLE["accent2"], STYLE["accent5"], STYLE["accent6"],
]

# Per-model palette (8 models, one color each)
MODEL_PALETTE: list[str] = [
    "#2563eb", "#f59e0b", "#10b981", "#ef4444",
    "#8b5cf6", "#06b6d4", "#f97316", "#64748b",
]

# Per-PCA-component colors (up to 10 components)
PCA_COLORS: list[str] = [
    "#e63946", "#2a9d8f", "#e9c46a", "#457b9d", "#f4a261",
    "#6a4c93", "#1982c4", "#8ac926", "#ff595e", "#ffca3a",
]


# ---------------------------------------------------------------------------
# Matplotlib theme — call once at script startup
# ---------------------------------------------------------------------------
def _apply_style() -> None:
    for s in ["seaborn-v0_8-whitegrid", "seaborn-whitegrid", "ggplot"]:
        try:
            plt.style.use(s)
            break
        except Exception:
            continue
    plt.rcParams.update({
        "font.family":       "DejaVu Sans",
        "font.size":         11,
        "axes.facecolor":    STYLE["panel"],
        "figure.facecolor":  STYLE["bg"],
        "axes.grid":         True,
        "grid.alpha":        0.4,
        "grid.linewidth":    0.5,
        "axes.spines.top":   False,
        "axes.spines.right": False,
    })
