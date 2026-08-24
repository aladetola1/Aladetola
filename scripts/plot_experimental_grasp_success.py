from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Experimental grasp-success data

RANKS = np.arange(1, 11)

SUCCESS_DATA = {
    "Mug": [100, 100, 100, 100, 90, 90, 80, 80, 80, 80],
    "Bowl": [100, 100, 100, 100, 100, 100, 90, 90, 90, 90],
    "Spoon": [100, 100, 100, 100, 100, 90, 90, 90, 80, 80],
}

OUTPUT_DIR = Path("results/experimental_validation")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def configure_axis(ax: plt.Axes, title: str) -> None:
    """Apply consistent manuscript formatting."""
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("GFI Rank", fontsize=10)
    ax.set_ylabel("Grasp Success Rate (%)", fontsize=10)

    ax.set_xticks(RANKS)
    ax.set_ylim(0, 110)
    ax.set_yticks(np.arange(0, 101, 20))

    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.7)
    ax.tick_params(axis="both", labelsize=9)


def remove_old_outputs() -> None:
    """Delete obsolete files so they are not mistaken for current results."""
    old_files = [
        OUTPUT_DIR / "screwdriver_grasp_success.pdf",
        OUTPUT_DIR / "screwdriver_grasp_success.png",
        OUTPUT_DIR / "grasp_success_by_rank.pdf",
        OUTPUT_DIR / "grasp_success_by_rank.png",
    ]

    for path in old_files:
        if path.exists():
            path.unlink()
            print(f"Deleted old file: {path}")


def save_individual_bar_plots() -> None:
    """Generate one bar chart for each experimental object."""
    for object_name, success_rates in SUCCESS_DATA.items():
        fig, ax = plt.subplots(figsize=(5.0, 3.5))

        bars = ax.bar(RANKS, success_rates, width=0.72)

        configure_axis(
            ax,
            f"{object_name}: Grasp Success Across Top-10 GFI Placements",
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
            OUTPUT_DIR / f"{filename}_grasp_success.pdf",
            bbox_inches="tight",
            pad_inches=0.02,
        )
        fig.savefig(
            OUTPUT_DIR / f"{filename}_grasp_success.png",
            dpi=600,
            bbox_inches="tight",
            pad_inches=0.02,
        )

        plt.close(fig)


def save_combined_line_plot() -> None:
    """Generate one comparison plot for Mug, Bowl, and Spoon."""

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

    # Axis labels
    ax.set_xlabel("GFI Rank", fontsize=10)
    ax.set_ylabel("Grasp Success Rate (%)", fontsize=10)

    # Top-10 GFI ranks
    ax.set_xticks(RANKS)

    ax.set_ylim(60, 101.5)
    ax.set_yticks([60, 70, 80, 90, 100])

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

    # Legend
    ax.legend(
        fontsize=9,
        frameon=True,
        loc="upper right",
    )

    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR / "combined_grasp_success_comparison.pdf",
        bbox_inches="tight",
        pad_inches=0.08,
    )

    fig.savefig(
        OUTPUT_DIR / "combined_grasp_success_comparison.png",
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.08,
    )

    fig.savefig(
        OUTPUT_DIR / "grasp_success_by_rank.pdf",
        bbox_inches="tight",
        pad_inches=0.08,
    )

    fig.savefig(
        OUTPUT_DIR / "grasp_success_by_rank.png",
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.08,
    )

    plt.close(fig)

def save_average_success_plot() -> None:
    """Plot the mean success rate across the ten ranked placements."""
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
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.7)
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
        OUTPUT_DIR / "mean_grasp_success_by_object.pdf",
        bbox_inches="tight",
        pad_inches=0.02,
    )
    fig.savefig(
        OUTPUT_DIR / "mean_grasp_success_by_object.png",
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.02,
    )

    plt.close(fig)


def export_data_table() -> None:
    """Export the experimental data and calculated averages."""
    rows = []

    for object_name, success_rates in SUCCESS_DATA.items():
        for rank, success_rate in zip(RANKS, success_rates):
            rows.append(
                {
                    "object": object_name,
                    "gfi_rank": int(rank),
                    "grasp_success_rate_percent": float(success_rate),
                }
            )

    dataframe = pd.DataFrame(rows)

    dataframe.to_csv(
        OUTPUT_DIR / "experimental_grasp_success_data.csv",
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
        OUTPUT_DIR / "experimental_grasp_success_summary.csv",
        index=False,
    )

    print("\nMean grasp success rate:")
    print(summary.to_string(index=False))


def main() -> None:
    remove_old_outputs()
    export_data_table()
    save_individual_bar_plots()
    save_combined_line_plot()
    save_average_success_plot()

    print("\nFigures saved in:")
    print(OUTPUT_DIR.resolve())


if __name__ == "__main__":
    main()
