"""
utils/chart_utils.py
====================
Generate confidence analysis charts from recognition_tests table.
Run standalone:
    python utils/chart_utils.py

Or call generate_all_charts() from any script to save PNGs to data/charts/.

Charts produced
---------------
1. confidence_histogram.png
   Distribution of cosine-similarity scores (green = correct, red = incorrect).

2. confidence_over_time.png
   Line chart of running-average confidence across consecutive tests.

3. accuracy_by_band.png
   Bar chart: accuracy (%) inside each confidence band
   [0-0.5), [0.5-0.6), [0.6-0.7), [0.7-0.8), [0.8-0.9), [0.9-1.0]

4. person_confidence_boxplot.png
   Box plot per person showing confidence spread (top-20 by test count).

Model pipeline reminder (inline comment)
-----------------------------------------
  Frame (BGR)
    └─► SCRFD (detection)     → bbox [x1,y1,x2,y2]  + 5 landmarks
          └─► align crop 112×112
                └─► ResNet-50 (backbone)
                      Feature map [B, 2048, 7, 7]
                      └─► Global Average Pool  [B, 2048]
                            └─► ArcFace head
                                  FC 2048→512  +  BN  +  L2-norm
                                  → Embedding [B, 512]  ‖e‖=1
                                        └─► Cosine sim vs gallery
                                              ≥0.70 → confident
                                              0.50–0.69 → uncertain
                                              <0.50 → unknown
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless – no display needed
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

CHART_DIR = os.path.join(ROOT, "data", "charts")
os.makedirs(CHART_DIR, exist_ok=True)

# ── theme ─────────────────────────────────────────────────────────────────────
BG    = "#0f1117"
CARD  = "#1a1d27"
TEXT  = "#e8eaf0"
MUTED = "#8891a8"
GREEN = "#3ddc84"
RED   = "#f7604f"
AMBER = "#f7c44f"
BLUE  = "#4f8ef7"

plt.rcParams.update({
    "figure.facecolor" : BG,
    "axes.facecolor"   : CARD,
    "axes.edgecolor"   : "#2e3248",
    "axes.labelcolor"  : TEXT,
    "xtick.color"      : MUTED,
    "ytick.color"      : MUTED,
    "text.color"       : TEXT,
    "grid.color"       : "#2e3248",
    "grid.linewidth"   : 0.6,
    "font.family"      : "DejaVu Sans",
    "font.size"        : 10,
})


def _load_data():
    """Fetch all recognition_tests rows from DB."""
    from models.attendance_model import get_confidence_history
    rows = get_confidence_history(limit=5000)
    confs     = np.array([r["confidence"]  for r in rows], dtype=float)
    correct   = np.array([bool(r["is_correct"]) for r in rows])
    names     = [r["name"] or "Unknown"    for r in rows]
    tested_at = [r["tested_at"]            for r in rows]
    return confs, correct, names, tested_at, rows


# ── Chart 1 : Histogram ───────────────────────────────────────────────────────
def chart_histogram(confs, correct, path):
    fig, ax = plt.subplots(figsize=(8, 4))
    bins = np.linspace(0, 1, 21)
    ax.hist(confs[correct],  bins=bins, color=GREEN, alpha=0.75, label="Correct")
    ax.hist(confs[~correct], bins=bins, color=RED,   alpha=0.75, label="Incorrect")
    ax.axvline(0.70, color=GREEN, ls="--", lw=1.2, label="Confident threshold (0.70)")
    ax.axvline(0.50, color=AMBER, ls="--", lw=1.2, label="Uncertain threshold (0.50)")
    ax.set_xlabel("Cosine Similarity (Confidence)")
    ax.set_ylabel("Count")
    ax.set_title("Confidence Score Distribution", color=TEXT, fontsize=12, pad=10)
    ax.legend(facecolor=CARD, edgecolor="#2e3248", labelcolor=TEXT, fontsize=9)
    ax.grid(True, axis="y")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"[chart] {path}")


# ── Chart 2 : Confidence over time ───────────────────────────────────────────
def chart_over_time(confs, path, window=20):
    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(len(confs))
    ax.plot(x, confs, color=MUTED, lw=0.8, alpha=0.5, label="Per-test")
    if len(confs) >= window:
        running = np.convolve(confs, np.ones(window)/window, mode="valid")
        ax.plot(np.arange(window-1, len(confs)), running,
                color=BLUE, lw=2, label=f"Moving avg (n={window})")
    ax.axhline(0.70, color=GREEN, ls="--", lw=1, alpha=0.7)
    ax.axhline(0.50, color=AMBER, ls="--", lw=1, alpha=0.7)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Test index")
    ax.set_ylabel("Confidence")
    ax.set_title("Confidence Over Time", color=TEXT, fontsize=12, pad=10)
    ax.legend(facecolor=CARD, edgecolor="#2e3248", labelcolor=TEXT, fontsize=9)
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"[chart] {path}")


# ── Chart 3 : Accuracy by confidence band ────────────────────────────────────
def chart_accuracy_bands(confs, correct, path):
    bands  = [(0.0,0.5),(0.5,0.6),(0.6,0.7),(0.7,0.8),(0.8,0.9),(0.9,1.01)]
    labels = ["<0.50","0.50–0.60","0.60–0.70","0.70–0.80","0.80–0.90","≥0.90"]
    accs, counts = [], []
    for lo, hi in bands:
        mask = (confs >= lo) & (confs < hi)
        cnt  = mask.sum()
        acc  = correct[mask].mean() * 100 if cnt > 0 else 0
        accs.append(acc)
        counts.append(cnt)

    fig, ax = plt.subplots(figsize=(9, 4))
    colors = [RED, RED, AMBER, AMBER, GREEN, GREEN]
    bars = ax.bar(labels, accs, color=colors, alpha=0.85, width=0.55)
    for bar, cnt, acc in zip(bars, counts, accs):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1,
                f"n={cnt}\n{acc:.0f}%", ha="center", va="bottom",
                fontsize=8, color=TEXT)
    ax.set_ylim(0, 115)
    ax.set_xlabel("Confidence Band")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Accuracy by Confidence Band", color=TEXT, fontsize=12, pad=10)
    ax.grid(True, axis="y")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"[chart] {path}")


# ── Chart 4 : Box plot per person ─────────────────────────────────────────────
def chart_boxplot(confs, names, path, top_n=20):
    from collections import defaultdict
    person_confs = defaultdict(list)
    for c, n in zip(confs, names):
        person_confs[n].append(c)

    # Keep top_n by count
    sorted_people = sorted(person_confs.items(), key=lambda x: -len(x[1]))[:top_n]
    people  = [p for p, _ in sorted_people]
    data    = [person_confs[p] for p in people]
    labels  = [p[:16] for p in people]

    fig, ax = plt.subplots(figsize=(max(10, len(people)*0.6), 5))
    bp = ax.boxplot(data, patch_artist=True, notch=False,
                    medianprops=dict(color=TEXT, lw=1.5),
                    flierprops=dict(marker=".", color=MUTED, markersize=3))
    for patch in bp["boxes"]:
        patch.set_facecolor(BLUE)
        patch.set_alpha(0.6)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.axhline(0.70, color=GREEN, ls="--", lw=1, alpha=0.7, label="0.70")
    ax.axhline(0.50, color=AMBER, ls="--", lw=1, alpha=0.7, label="0.50")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Confidence")
    ax.set_title(f"Confidence Distribution per Person (top {top_n})",
                 color=TEXT, fontsize=12, pad=10)
    ax.legend(facecolor=CARD, edgecolor="#2e3248", labelcolor=TEXT, fontsize=9)
    ax.grid(True, axis="y")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"[chart] {path}")


# ── Composite dashboard chart ────────────────────────────────────────────────
def chart_dashboard(confs, correct, names, path, window=20):
    """2×2 grid combining all 4 charts into one file."""
    fig = plt.figure(figsize=(16, 10))
    gs  = GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.32)

    # 1 – histogram
    ax1 = fig.add_subplot(gs[0, 0])
    bins = np.linspace(0, 1, 21)
    ax1.hist(confs[correct],  bins=bins, color=GREEN, alpha=0.75, label="Correct")
    ax1.hist(confs[~correct], bins=bins, color=RED,   alpha=0.75, label="Incorrect")
    ax1.axvline(0.70, color=GREEN, ls="--", lw=1)
    ax1.axvline(0.50, color=AMBER, ls="--", lw=1)
    ax1.set_title("Score Distribution", color=TEXT)
    ax1.legend(facecolor=CARD, edgecolor="#2e3248", labelcolor=TEXT, fontsize=8)
    ax1.grid(True, axis="y")

    # 2 – over time
    ax2 = fig.add_subplot(gs[0, 1])
    x = np.arange(len(confs))
    ax2.plot(x, confs, color=MUTED, lw=0.7, alpha=0.45)
    if len(confs) >= window:
        running = np.convolve(confs, np.ones(window)/window, mode="valid")
        ax2.plot(np.arange(window-1, len(confs)), running, color=BLUE, lw=2)
    ax2.axhline(0.70, color=GREEN, ls="--", lw=0.9)
    ax2.axhline(0.50, color=AMBER, ls="--", lw=0.9)
    ax2.set_ylim(0, 1.05)
    ax2.set_title("Confidence Over Time", color=TEXT)
    ax2.grid(True)

    # 3 – accuracy bands
    ax3 = fig.add_subplot(gs[1, 0])
    bands  = [(0,0.5),(0.5,0.6),(0.6,0.7),(0.7,0.8),(0.8,0.9),(0.9,1.01)]
    blabels = ["<.50",".50–.60",".60–.70",".70–.80",".80–.90","≥.90"]
    accs   = []
    for lo, hi in bands:
        mask = (confs >= lo) & (confs < hi)
        accs.append(correct[mask].mean()*100 if mask.sum() else 0)
    bcols = [RED, RED, AMBER, AMBER, GREEN, GREEN]
    ax3.bar(blabels, accs, color=bcols, alpha=0.85, width=0.6)
    ax3.set_ylim(0, 115)
    ax3.set_title("Accuracy by Band", color=TEXT)
    ax3.grid(True, axis="y")
    ax3.tick_params(axis="x", labelsize=8)

    # 4 – summary stats text
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis("off")
    total     = len(confs)
    n_correct = correct.sum()
    acc_total = n_correct/total*100 if total else 0
    high_mask = confs >= 0.70
    n_high    = high_mask.sum()
    acc_high  = correct[high_mask].mean()*100 if n_high else 0
    lines = [
        ("Total tests",        f"{total}"),
        ("Overall accuracy",   f"{acc_total:.1f}%"),
        ("Confident (≥0.70)", f"{n_high} ({n_high/total*100:.1f}%)" if total else "—"),
        ("Acc @ confident",    f"{acc_high:.1f}%"),
        ("Mean confidence",    f"{confs.mean():.3f}" if total else "—"),
        ("Std confidence",     f"{confs.std():.3f}"  if total else "—"),
    ]
    for i, (lbl, val) in enumerate(lines):
        ax4.text(0.05, 0.88-i*0.15, lbl, transform=ax4.transAxes,
                 color=MUTED, fontsize=10)
        ax4.text(0.60, 0.88-i*0.15, val, transform=ax4.transAxes,
                 color=TEXT, fontsize=11, fontweight="bold")
    ax4.set_title("Summary Statistics", color=TEXT)

    fig.suptitle("FRA – Recognition Confidence Dashboard",
                 color=TEXT, fontsize=14, y=0.99)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[chart] {path}")


# ── Entry point ───────────────────────────────────────────────────────────────
def generate_all_charts():
    confs, correct, names, tested_at, _ = _load_data()
    if len(confs) == 0:
        print("[chart] No data in recognition_tests yet. Run some tests first.")
        return

    chart_histogram   (confs, correct, os.path.join(CHART_DIR, "confidence_histogram.png"))
    chart_over_time   (confs,          os.path.join(CHART_DIR, "confidence_over_time.png"))
    chart_accuracy_bands(confs, correct,os.path.join(CHART_DIR, "accuracy_by_band.png"))
    chart_boxplot     (confs, names,   os.path.join(CHART_DIR, "person_confidence_boxplot.png"))
    chart_dashboard   (confs, correct, names,
                                       os.path.join(CHART_DIR, "dashboard.png"))
    print(f"\n[chart] All charts saved to {CHART_DIR}")


if __name__ == "__main__":
    from database.db import init_db
    init_db()
    generate_all_charts()
