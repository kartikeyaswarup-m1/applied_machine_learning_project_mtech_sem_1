"""
Shared configuration: paths, plotting style, and a helper for saving figures.
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")  # headless backend so it runs anywhere (CI, servers)
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
IMAGES_DIR = ROOT_DIR / "images"
RAW_XLSX = DATA_DIR / "online_retail_II.xlsx"
CACHE_FILE = DATA_DIR / "clean_retail.pkl"
SAMPLE_CSV = DATA_DIR / "sample_data.csv"

DATA_DIR.mkdir(exist_ok=True)
IMAGES_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Plotting style — consistent, clean look across every chart
# ---------------------------------------------------------------------------
PALETTE = ["#2E5EAA", "#5BC0BE", "#F4A259", "#E76F51", "#8338EC",
           "#06A77D", "#D62246", "#3A86FF", "#FFBE0B", "#264653"]


def set_style():
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 120,
        "figure.autolayout": True,
        "axes.titleweight": "bold",
        "axes.titlesize": 15,
        "axes.labelsize": 12,
        "axes.edgecolor": "#444444",
        "axes.prop_cycle": plt.cycler(color=PALETTE),
        "font.size": 11,
        "legend.frameon": False,
    })


def save_fig(fig, name: str):
    """Save a figure into images/ and close it."""
    set_style.called = True
    out = IMAGES_DIR / name
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [image] saved {out.relative_to(ROOT_DIR)}")
    return out
