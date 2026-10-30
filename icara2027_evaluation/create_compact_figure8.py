from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# Compact side-by-side Figure 8 for ICARA 2027
# ------------------------------------------------------------

output_pdf = Path("ICARA2027_GFI_vs_Panda_COMPACT_SIDE_BY_SIDE.pdf")
output_png = Path("ICARA2027_GFI_vs_Panda_COMPACT_SIDE_BY_SIDE.png")

# Panel (a): AUROC values across all placements
objects = ["Banana", "Power Drill", "Mug", "Spoon", "Bowl"]
gfi = np.array([0.560, 1.000, 0.489, 0.500, 0.517])
panda = np.array([0.792, 0.912, 0.970, 0.455, 0.500])

# Panel (b): paired differences = GFI - Panda
labels_b = [
    "Banana (21/29)",
    "Power Drill (34/35)",
    "Mug (22/34)",
    "Spoon (22/24)",
    "Bowl (2/32)",
]
delta = np.array([-0.232, 0.088, -0.481, 0.045, 0.017])
ci_low = np.array([-0.595, 0.000, -0.739, -0.409, -0.233])
ci_high = np.array([0.131, 0.206, -0.227, 0.500, 0.267])

# Convert CI bounds to asymmetric error-bar distances
xerr = np.vstack([delta - ci_low, ci_high - delta])

# Figure layout
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
ax1, ax2 = axes

# ---------------- Panel (a) ----------------
x = np.arange(len(objects))
width = 0.34

bars1 = ax1.bar(x - width/2, gfi, width, label="Robot-independent GFI")
bars2 = ax1.bar(x + width/2, panda, width, label="Panda-specific baseline")

ax1.axhline(0.5, linestyle=":", linewidth=1.2, label="Chance level")
ax1.set_ylim(0, 1.15)
ax1.set_ylabel("AUROC")
ax1.set_xticks(x)
ax1.set_xticklabels(objects)
ax1.set_title("(a) AUROC for grasp-execution success", fontweight="bold", pad=10)

# Bar labels
for bars in (bars1, bars2):
    for bar in bars:
        h = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width()/2,
            h + 0.02,
            f"{h:.3f}",
            ha="center",
            va="bottom",
            fontsize=8
        )

ax1.legend(loc="upper center", bbox_to_anchor=(0.5, 1.18), ncol=3, frameon=False, fontsize=8)

# ---------------- Panel (b) ----------------
y = np.arange(len(labels_b))[::-1]

ax2.errorbar(
    delta, y,
    xerr=xerr,
    fmt='o',
    capsize=4,
    linewidth=1.5
)

ax2.axvline(0, linestyle="--", linewidth=1.0)
ax2.set_yticks(y)
ax2.set_yticklabels(labels_b)
ax2.set_xlim(-0.85, 0.60)
ax2.set_xlabel("AUROC difference: GFI - Panda (95% bootstrap CI)")
ax2.set_title("(b) Paired AUROC differences", fontweight="bold", pad=10)
ax2.grid(axis="x", alpha=0.25)

# Footer note
fig.text(
    0.5, 0.01,
    "Five selected YCB objects; 154 ranking-selected evaluated placements. "
    "5,000 paired class-stratified bootstrap resamples.",
    ha="center",
    va="bottom",
    fontsize=8
)

plt.tight_layout(rect=[0, 0.05, 1, 1])

fig.savefig(output_pdf, dpi=300, bbox_inches="tight")
fig.savefig(output_png, dpi=300, bbox_inches="tight")
plt.close(fig)

print("Saved:", output_pdf.resolve())
print("Saved:", output_png.resolve())
