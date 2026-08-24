from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# Mean grasp-success data
# ============================================================

OBJECTS = ["Mug", "Bowl", "Spoon"]

RGMP_MEANS = [90.0, 96.0, 93.0]
RANDOM_MEANS = [36.0, 47.0, 43.0]

OUTPUT_DIR = Path(
    "results/experimental_validation/rgmp_vs_random"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Plot
# ============================================================

def save_comparison_plot() -> None:
    """Compare RGMP-ranked placement selection with random selection."""

    x = np.arange(len(OBJECTS))
    width = 0.34

    fig, ax = plt.subplots(figsize=(6.2, 4.0))

    bars_rgmp = ax.bar(
        x - width / 2,
        RGMP_MEANS,
        width,
        label="RGMP-ranked",
    )

    bars_random = ax.bar(
        x + width / 2,
        RANDOM_MEANS,
        width,
        label="Random selection",
    )

    ax.set_xlabel(
        "Experimental Object",
        fontsize=10,
    )

    ax.set_ylabel(
        "Mean Grasp Success Rate (%)",
        fontsize=10,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(OBJECTS)

    ax.set_ylim(0, 110)
    ax.set_yticks(np.arange(0, 101, 20))

    ax.grid(
        axis="y",
        linestyle="--",
        linewidth=0.6,
        alpha=0.7,
    )

    ax.tick_params(
        axis="both",
        labelsize=9,
    )

    ax.legend(
        fontsize=9,
        frameon=True,
        loc="upper right",
    )


    for bars in [bars_rgmp, bars_random]:
        for bar in bars:
            value = bar.get_height()

            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 1.5,
                f"{value:.0f}%",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR / "rgmp_vs_random_mean_grasp_success.pdf",
        bbox_inches="tight",
        pad_inches=0.04,
    )

    fig.savefig(
        OUTPUT_DIR / "rgmp_vs_random_mean_grasp_success.png",
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.04,
    )

    plt.close(fig)


# ============================================================
# Numerical summary
# ============================================================

def print_summary() -> None:
    """Print the comparison values and improvement."""

    print("\nRGMP vs Random Mean Grasp Success\n")

    for object_name, rgmp, random in zip(
        OBJECTS,
        RGMP_MEANS,
        RANDOM_MEANS,
    ):
        difference = rgmp - random

        print(
            f"{object_name:6s}: "
            f"RGMP = {rgmp:.1f}% | "
            f"Random = {random:.1f}% | "
            f"Difference = {difference:.1f} percentage points"
        )

    rgmp_overall = float(np.mean(RGMP_MEANS))
    random_overall = float(np.mean(RANDOM_MEANS))
    overall_difference = rgmp_overall - random_overall

    print("\nOverall mean:")
    print(f"RGMP   = {rgmp_overall:.1f}%")
    print(f"Random = {random_overall:.1f}%")
    print(
        f"Difference = "
        f"{overall_difference:.1f} percentage points"
    )


# ============================================================
# Main
# ============================================================

def main() -> None:
    save_comparison_plot()
    print_summary()

    print("\nComparison figure saved in:")
    print(OUTPUT_DIR.resolve())


if __name__ == "__main__":
    main()
