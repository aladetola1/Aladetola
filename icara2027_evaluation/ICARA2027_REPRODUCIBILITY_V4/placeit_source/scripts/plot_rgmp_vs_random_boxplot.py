from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42



# ============================================================
# Experimental grasp-success data across 10 placements
# ============================================================

RGMP_DATA = {
    "Mug": [
        100, 100, 100, 100, 90,
        90, 80, 80, 80, 80
    ],
    "Bowl": [
        100, 100, 100, 100, 100,
        100, 90, 90, 90, 90
    ],
    "Spoon": [
        100, 100, 100, 100, 100,
        90, 90, 90, 80, 80
    ],
}

RANDOM_DATA = {
    "Mug": [
        100, 100, 100, 0, 0,
        0, 0, 0, 60, 0
    ],
    "Bowl": [
        100, 100, 100, 100, 0,
        70, 0, 0, 0, 0
    ],
    "Spoon": [
        100, 100, 70, 60, 60,
        0, 0, 0, 40, 0
    ],
}

OBJECTS = ["Mug", "Bowl", "Spoon"]

OUTPUT_DIR = Path(
    "results/experimental_validation/rgmp_vs_random"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def save_boxplot() -> None:

    fig, ax = plt.subplots(figsize=(6.8, 4.3))

    positions_rgmp = [0.8, 2.8, 4.8]
    positions_random = [1.2, 3.2, 5.2]

    rgmp_values = [
        RGMP_DATA[obj]
        for obj in OBJECTS
    ]

    random_values = [
        RANDOM_DATA[obj]
        for obj in OBJECTS
    ]

    # RGMP-ranked distributions
    bp_rgmp = ax.boxplot(
        rgmp_values,
        positions=positions_rgmp,
        widths=0.32,
        patch_artist=True,
        showmeans=True,
        meanline=False,
        showfliers=True,
    )

    # Random-selection distributions
    bp_random = ax.boxplot(
        random_values,
        positions=positions_random,
        widths=0.32,
        patch_artist=True,
        showmeans=True,
        meanline=False,
        showfliers=True,
    )

    # Use Matplotlib default colors
    default_colors = plt.rcParams[
        "axes.prop_cycle"
    ].by_key()["color"]

    for box in bp_rgmp["boxes"]:
        box.set_facecolor(default_colors[0])
        box.set_alpha(0.75)

    for box in bp_random["boxes"]:
        box.set_facecolor(default_colors[1])
        box.set_alpha(0.75)

    # Make median lines visible
    for median in bp_rgmp["medians"]:
        median.set_linewidth(1.6)

    for median in bp_random["medians"]:
        median.set_linewidth(1.6)

    # Overlay individual observations
    for i, obj in enumerate(OBJECTS):

        ax.scatter(
            np.full(
                len(RGMP_DATA[obj]),
                positions_rgmp[i]
            ),
            RGMP_DATA[obj],
            s=18,
            alpha=0.55,
            zorder=3,
        )

        ax.scatter(
            np.full(
                len(RANDOM_DATA[obj]),
                positions_random[i]
            ),
            RANDOM_DATA[obj],
            s=18,
            alpha=0.55,
            zorder=3,
        )

    ax.set_xticks([1, 3, 5])
    ax.set_xticklabels(OBJECTS)

    ax.set_xlabel(
        "Experimental Object",
        fontsize=10,
    )

    ax.set_ylabel(
        "Grasp Success Rate (%)",
        fontsize=10,
    )

    ax.set_ylim(-5, 110)
    ax.set_yticks(
        np.arange(0, 101, 20)
    )

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

    # Legend
    handles = [
        plt.Rectangle(
            (0, 0),
            1,
            1,
            facecolor=default_colors[0],
            alpha=0.75,
            label="RGMP-ranked",
        ),
        plt.Rectangle(
            (0, 0),
            1,
            1,
            facecolor=default_colors[1],
            alpha=0.75,
            label="Random selection",
        ),
    ]

    ax.legend(
        handles=handles,
        fontsize=9,
        loc="upper right",
        frameon=True,
    )

    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR /
        "rgmp_vs_random_grasp_success_boxplot.pdf",
        bbox_inches="tight",
        pad_inches=0.04,
    )

    fig.savefig(
        OUTPUT_DIR /
        "rgmp_vs_random_grasp_success_boxplot.png",
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.04,
    )

    plt.close(fig)


def print_statistics() -> None:

    print("\nRGMP vs Random Distribution Statistics\n")

    for obj in OBJECTS:

        rgmp = np.asarray(
            RGMP_DATA[obj],
            dtype=float,
        )

        random = np.asarray(
            RANDOM_DATA[obj],
            dtype=float,
        )

        print(obj)

        print(
            f"  RGMP   : "
            f"mean={np.mean(rgmp):.1f}% | "
            f"median={np.median(rgmp):.1f}% | "
            f"std={np.std(rgmp):.1f}%"
        )

        print(
            f"  Random : "
            f"mean={np.mean(random):.1f}% | "
            f"median={np.median(random):.1f}% | "
            f"std={np.std(random):.1f}%"
        )

        print()


def main() -> None:

    save_boxplot()
    print_statistics()

    print("Box plot saved in:")
    print(OUTPUT_DIR.resolve())


if __name__ == "__main__":
    main()
