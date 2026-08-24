from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Random-selection grasp-success data
# 10 grasp attempts per rank

RANKS = np.arange(1, 11)

SUCCESS_DATA = {
    "Mug": [100, 100, 100, 0, 0, 0, 0, 0, 60, 0],
    "Bowl": [100, 100, 100, 100, 0, 70, 0, 0, 0, 0],
    "Spoon": [100, 100, 70, 60, 60, 0, 0, 0, 40, 0],
}

OUTPUT_DIR = Path(
    "results/experimental_validation/random_baseline"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def configure_axis(ax: plt.Axes, title: str) -> None:
    """Apply consistent manuscript formatting."""
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Random Placement Sample", fontsize=10)
    ax.set_ylabel("Grasp Success Rate (%)", fontsize=10)

    ax.set_xticks(RANKS)
    ax.set_ylim(0, 110)
    ax.set_yticks(np.arange(0, 101, 20))

    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.7)
    ax.tick_params(axis="both", labelsize=9)


def save_individual_bar_plots() -> None:
    """Generate one bar chart for each experimental object."""
    for object_name, success_rates in SUCCESS_DATA.items():
        fig, ax = plt.subplots(figsize=(5.0, 3.5))

        bars = ax.bar(RANKS, success_rates, width=0.72)

        configure_axis(
            ax,
            f"{object_name}: Random-Selection Grasp Success",
        )

        for bar, value in zip(bars, success_rates):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 1.5,
                f"{value}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

        fig.tight_layout()

        filename = object_name.lower().replace(" ", "_")

        fig.savefig(
            OUTPUT_DIR / f"{filename}_random_grasp_success.pdf",
            bbox_inches="tight",
            pad_inches=0.02,
        )

        fig.savefig(
            OUTPUT_DIR / f"{filename}_random_grasp_success.png",
            dpi=600,
            bbox_inches="tight",
            pad_inches=0.02,
        )

        plt.close(fig)


def save_combined_line_plot() -> None:
    """Generate combined random-selection comparison plot."""

    fig, ax = plt.subplots(figsize=(6.8, 4.2))

    marker_styles = {
        "Mug": "o",
        "Bowl": "s",
        "Spoon": "^",
    }

    line_styles = {
        "Mug": "-",
        "Bowl": "--",
        "Spoon": "-.",
    }

    for object_name, success_rates in SUCCESS_DATA.items():
        ax.plot(
            RANKS,
            success_rates,
            marker=marker_styles[object_name],
            linestyle=line_styles[object_name],
            linewidth=1.7,
            markersize=5,
            label=object_name,
        )

    ax.set_xlabel("Random Placement Sample", fontsize=10)
    ax.set_ylabel("Grasp Success Rate (%)", fontsize=10)

    ax.set_xticks(RANKS)

    ax.set_ylim(0, 105)
    ax.set_yticks([0, 20, 40, 60, 80, 100])

    ax.set_xlim(0.55, 10.45)

    ax.grid(
        True,
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

    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR / "random_grasp_success_by_rank.pdf",
        bbox_inches="tight",
        pad_inches=0.08,
    )

    fig.savefig(
        OUTPUT_DIR / "random_grasp_success_by_rank.png",
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.08,
    )

    plt.close(fig)


def save_average_success_plot() -> None:
    """Plot mean random-selection grasp success."""
    object_names = list(SUCCESS_DATA.keys())

    averages = [
        float(np.mean(SUCCESS_DATA[name]))
        for name in object_names
    ]

    fig, ax = plt.subplots(figsize=(5.5, 3.7))

    bars = ax.bar(object_names, averages, width=0.65)

    ax.set_xlabel("Experimental Object", fontsize=10)
    ax.set_ylabel("Mean Grasp Success Rate (%)", fontsize=10)

    ax.set_ylim(0, 110)
    ax.set_yticks(np.arange(0, 101, 20))

    ax.grid(
        axis="y",
        linestyle="--",
        linewidth=0.6,
        alpha=0.7,
    )

    ax.tick_params(axis="both", labelsize=9)

    for bar, value in zip(bars, averages):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 1.5,
            f"{value:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR / "random_mean_grasp_success_by_object.pdf",
        bbox_inches="tight",
        pad_inches=0.02,
    )

    fig.savefig(
        OUTPUT_DIR / "random_mean_grasp_success_by_object.png",
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.02,
    )

    plt.close(fig)


def export_data_table() -> None:
    """Export random-selection data and calculated averages."""
    rows = []

    for object_name, success_rates in SUCCESS_DATA.items():
        for rank, success_rate in zip(RANKS, success_rates):
            rows.append(
                {
                    "object": object_name,
                    "sample": int(rank),
                    "grasp_success_rate_percent": float(success_rate),
                }
            )

    dataframe = pd.DataFrame(rows)

    dataframe.to_csv(
        OUTPUT_DIR / "random_grasp_success_data.csv",
        index=False,
    )

    summary = (
        dataframe.groupby("object", as_index=False)[
            "grasp_success_rate_percent"
        ]
        .mean()
        .rename(
            columns={
                "grasp_success_rate_percent":
                "mean_grasp_success_rate_percent"
            }
        )
    )

    summary.to_csv(
        OUTPUT_DIR / "random_grasp_success_summary.csv",
        index=False,
    )

    print("\nMean random grasp success rate:")
    print(summary.to_string(index=False))


def main() -> None:
    export_data_table()
    save_individual_bar_plots()
    save_combined_line_plot()
    save_average_success_plot()

    print("\nRandom baseline figures saved in:")
    print(OUTPUT_DIR.resolve())


if __name__ == "__main__":
    main()
