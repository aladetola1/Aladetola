from pathlib import Path
import shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import rankdata, spearmanr

ROOT = Path.cwd()
OUT = ROOT / "FS_RGMP_RETROSPECTIVE" / "FOUR_OBJECT_FIGURES_V1"
OUT.mkdir(parents=True, exist_ok=True)

WINDOWS = Path(
    "/mnt/c/Users/olaol/Downloads/ICARA2027_GFI_FIGURES"
)

df = pd.read_csv(
    ROOT / "step43fr_verified_gfi_grasp_analysis.csv"
)

objects = ["banana", "mug", "spoon", "power_drill"]
names = ["Banana", "Mug", "Spoon", "Power Drill"]

PASS = {"SUCCESS", "APPROACH_FAILED", "GRASP_FAILED"}

df = df[
    df["object"].isin(objects)
    & df["status"].isin(PASS)
].copy()

assert not df.duplicated(["object", "archive_index"]).any()
assert len(df) == 111
assert int(df["success"].sum()) == 99

df["GFI_score"] = (
    0.5*df["H"] + 0.3*df["D"] + 0.2*df["U"]
)

assert np.allclose(
    df["GFI_score"], df["GFI"],
    atol=1e-8, rtol=1e-7
)

def auc(y, scores):
    y = np.asarray(y, dtype=int)
    scores = np.asarray(scores, dtype=float)

    n1 = int(y.sum())
    n0 = len(y) - n1

    if not n1 or not n0:
        return np.nan

    ranks = rankdata(scores, method="average")
    return float(
        (ranks[y == 1].sum() - n1*(n1+1)/2)
        / (n1*n0)
    )

def rho(y, scores):
    result = spearmanr(scores, y).statistic
    return float(result) if np.isfinite(result) else np.nan

def paired_ci(y, gfi, centrality, metric, seed):
    rng = np.random.default_rng(seed)
    y = np.asarray(y, dtype=int)
    gfi = np.asarray(gfi, dtype=float)
    centrality = np.asarray(centrality, dtype=float)

    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)

    differences = []

    # Paired class-stratified bootstrap:
    # Same sampled placements for both methods.
    for _ in range(3000):
        ids = np.concatenate([
            rng.choice(pos, len(pos), replace=True),
            rng.choice(neg, len(neg), replace=True)
        ])

        a = metric(y[ids], gfi[ids])
        b = metric(y[ids], centrality[ids])

        if np.isfinite(a) and np.isfinite(b):
            differences.append(a-b)

    return np.percentile(differences, [2.5, 97.5])

records = []

for i, obj in enumerate(objects):
    part = df[df["object"] == obj]

    y = part["success"].astype(int).to_numpy()
    g = part["GFI_score"].to_numpy()
    d = part["D"].to_numpy()

    assert len(np.unique(y)) == 2

    a_g = auc(y, g)
    a_d = auc(y, d)
    r_g = rho(y, g)
    r_d = rho(y, d)

    ci_auc = paired_ci(y, g, d, auc, 202700+i)
    ci_rho = paired_ci(y, g, d, rho, 202800+i)

    records.append({
        "object": obj,
        "n": len(part),
        "successes": int(y.sum()),
        "failures": int((y == 0).sum()),
        "gfi_auroc": a_g,
        "centrality_auroc": a_d,
        "auroc_difference": a_g-a_d,
        "auroc_ci_low": ci_auc[0],
        "auroc_ci_high": ci_auc[1],
        "gfi_spearman": r_g,
        "centrality_spearman": r_d,
        "spearman_difference": r_g-r_d,
        "spearman_ci_low": ci_rho[0],
        "spearman_ci_high": ci_rho[1],
    })

results = pd.DataFrame(records)

results.to_csv(
    OUT / "four_object_auroc_spearman_statistics.csv",
    index=False
)

print("\n===== VERIFIED FOUR-OBJECT STATISTICS =====")
print(results.round(4).to_string(index=False))

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.labelsize": 10,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

BLUE = "#1565C0"
RED = "#C62828"

def make_figure(metric, title, filename):
    gfi = results[f"gfi_{metric}"].to_numpy()
    cen = results[f"centrality_{metric}"].to_numpy()
    delta = results[f"{metric}_difference"].to_numpy()
    low = results[f"{metric}_ci_low"].to_numpy()
    high = results[f"{metric}_ci_high"].to_numpy()

    x = np.arange(len(objects))
    width = 0.35

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(8.0, 3.25),
        gridspec_kw={"width_ratios": [1.12, 1]}
    )

    ax1.bar(
        x-width/2, gfi, width,
        color=RED, label="GFI"
    )
    ax1.bar(
        x+width/2, cen, width,
        color=BLUE, label="Centrality-only"
    )

    ax1.set_xticks(x)
    ax1.set_xticklabels(
        ["Banana", "Mug", "Spoon", "Power\nDrill"]
    )
    ax1.set_ylabel(title)
    ax1.grid(axis="y", linestyle=":", alpha=0.3)
    ax1.set_axisbelow(True)
    ax1.legend(frameon=False, fontsize=8)

    if metric == "auroc":
        ax1.set_ylim(0, 1.05)
        ax1.axhline(
            0.5, linestyle=":", color="gray", linewidth=1
        )
    else:
        ax1.set_ylim(-1.05, 1.05)
        ax1.axhline(0, color="gray", linewidth=0.8)

    y = np.arange(len(objects))[::-1]

    ax2.axvline(0, color="gray", linestyle=":", linewidth=1)
    ax2.errorbar(
        delta, y,
        xerr=np.vstack([delta-low, high-delta]),
        fmt="o",
        color=RED,
        ecolor="black",
        capsize=3,
        markersize=5,
        linewidth=1.2
    )

    ax2.set_yticks(y)
    ax2.set_yticklabels(names)
    ax2.set_xlabel(
        "Difference: GFI - Centrality-only\n"
        "(95% paired bootstrap CI)"
    )
    ax2.grid(axis="x", linestyle=":", alpha=0.25)

    fig.subplots_adjust(
        left=0.09, right=0.98,
        top=0.94, bottom=0.26, wspace=0.45
    )

    for ext in ("pdf", "png", "svg"):
        path = OUT / f"{filename}.{ext}"
        fig.savefig(
            path,
            dpi=600 if ext == "png" else None,
            bbox_inches="tight",
            facecolor="white"
        )
        print("Saved:", path)

        if WINDOWS.is_dir():
            shutil.copy2(path, WINDOWS / path.name)
            print("Copied to Windows:", path.name)

    plt.close(fig)

make_figure(
    "auroc",
    "AUROC",
    "figure10_four_objects_panda_auroc"
)

make_figure(
    "spearman",
    "Spearman correlation",
    "figure12_four_objects_panda_spearman"
)

print("\nFIGURES 10 AND 12 COMPLETED")
print("No simulations rerun.")
print("Original TIAGo and Panda data unchanged.")
print("Four-object retrospective Panda analysis.")
