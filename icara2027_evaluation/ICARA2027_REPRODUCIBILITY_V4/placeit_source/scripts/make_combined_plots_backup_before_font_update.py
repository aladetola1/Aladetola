import os
import pickle
import shutil
import numpy as np
import matplotlib.pyplot as plt

RUNS = {
    "Mug": "results/mug/run87",
    "Bowl": "results/bowl/run89",
    "Spoon": "results/spoon/run94",
    "Power drill": "results/power_drill/run97",
    "Plate": "results/plate/run100",
    "Banana": "results/banana/run103",
}

OUT_DIR = "results/combined_plots"
FITNESS_ORIGINAL_DIR = os.path.join(
    OUT_DIR,
    "original_fitness_plots"
)

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(FITNESS_ORIGINAL_DIR, exist_ok=True)


def load_data(run_path):
    data_path = os.path.join(
        run_path,
        "data_export.pkl"
    )

    with open(data_path, "rb") as f:
        return pickle.load(f)


def load_final_archive(run_path):
    archive_path = os.path.join(
        run_path,
        "success_archives",
        "individuals_0.npz"
    )

    return np.load(
        archive_path,
        allow_pickle=True
    )


def add_zero_start(x, y):
    x = np.array(x)
    y = np.array(y)

    return (
        np.insert(x, 0, 0),
        np.insert(y, 0, 0)
    )


def save_line_plot(
    filename,
    title,
    ylabel,
    key,
    multiply_100=False
):
    plt.figure(figsize=(9, 6))

    for obj_name, run_path in RUNS.items():
        data = load_data(run_path)

        x = np.array(
            data["n_evals_hist"]
        )

        y = np.array(
            data[key]
        )

        if multiply_100:
            y = y * 100.0

        x, y = add_zero_start(x, y)

        plt.plot(
            x,
            y,
            linewidth=2,
            label=obj_name
        )

    plt.xlabel("Number of evaluations")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    save_path = os.path.join(
        OUT_DIR,
        filename
    )

    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("Saved:", save_path)


def plot_final_grasp_friendliness_boxplot():
    labels = []
    all_grasp_scores = []

    for obj_name, run_path in RUNS.items():
        archive = load_final_archive(run_path)

        infos = archive["infos"]

        grasp_scores = np.array(
            [x["grasp_proxy"] for x in infos],
            dtype=float
        )

        labels.append(obj_name)
        all_grasp_scores.append(grasp_scores)

        print(
            f"{obj_name}: "
            f"n={len(grasp_scores)}, "
            f"mean={np.mean(grasp_scores):.4f}, "
            f"median={np.median(grasp_scores):.4f}, "
            f"min={np.min(grasp_scores):.4f}, "
            f"max={np.max(grasp_scores):.4f}"
        )

    plt.figure(figsize=(9, 6))

    bp = plt.boxplot(
        all_grasp_scores,
        labels=labels,
        showmeans=True,
        patch_artist=False,
        meanprops=dict(
            marker="o",
            markerfacecolor="white",
            markeredgecolor="black",
            markersize=7
        ),
        medianprops=dict(
            linewidth=1.8
        )
    )

    # ---------------------------------------------------------
    # Legend: explicitly identify mean and median
    # ---------------------------------------------------------
    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markerfacecolor="white",
            markeredgecolor="black",
            markersize=7,
            label="Mean"
        ),
        Line2D(
            [0],
            [0],
            color="tab:orange",
            linewidth=1.8,
            label="Median"
        )
    ]
    plt.legend(
        handles=legend_elements,
        loc="upper right",
        ncol=2,
        frameon=True,
        columnspacing=1.2,
        handletextpad=0.5
    )

    plt.xlabel("Object")
    plt.ylabel("Grasp Friendliness Index (GFI)")
    plt.title("Distribution of GFI Across Stable Placements")

    plt.xticks(rotation=25)

    plt.grid(
        True,
        axis="y",
        alpha=0.3
    )

    plt.tight_layout()

    save_path = os.path.join(
        OUT_DIR,
        "combined_gfi_boxplot.png"
    )

    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("Saved:", save_path)


def plot_grasp_friendliness_histogram():
    plt.figure(figsize=(9, 6))

    for obj_name, run_path in RUNS.items():
        archive = load_final_archive(
            run_path
        )

        infos = archive["infos"]

        grasp_scores = np.array(
            [
                x["grasp_proxy"]
                for x in infos
            ],
            dtype=float
        )

        plt.hist(
            grasp_scores,
            bins=20,
            alpha=0.35,
            label=obj_name
        )

    plt.xlabel(
        "Grasp accessibility score"
    )

    plt.ylabel(
        "Number of placements"
    )

    plt.title(
        "Distribution of Grasp-Friendliness Across Objects"
    )

    plt.grid(
        True,
        alpha=0.3
    )

    plt.legend()
    plt.tight_layout()

    save_path = os.path.join(
        OUT_DIR,
        "combined_grasp_friendliness_histogram.png"
    )

    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("Saved:", save_path)


def plot_fitness_vs_grasp_scatter():
    plt.figure(figsize=(9, 6))

    for obj_name, run_path in RUNS.items():
        archive = load_final_archive(
            run_path
        )

        infos = archive["infos"]

        fitnesses = np.array(
            archive["fitnesses"],
            dtype=float
        )

        grasp_scores = np.array(
            [
                x["grasp_proxy"]
                for x in infos
            ],
            dtype=float
        )

        plt.scatter(
            fitnesses,
            grasp_scores,
            s=18,
            alpha=0.55,
            label=obj_name
        )

    plt.xlabel("Fitness")

    plt.ylabel(
        "Grasp accessibility score"
    )

    plt.title(
        "Fitness vs Grasp-Friendliness Across Objects"
    )

    plt.grid(
        True,
        alpha=0.3
    )

    plt.legend()
    plt.tight_layout()

    save_path = os.path.join(
        OUT_DIR,
        "combined_fitness_vs_grasp_scatter.png"
    )

    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("Saved:", save_path)


def copy_original_fitness_plots():
    for obj_name, run_path in RUNS.items():
        source_path = os.path.join(
            run_path,
            "plots",
            "fitness_of_successeful_individulas.png"
        )

        clean_name = (
            obj_name
            .lower()
            .replace(" ", "_")
        )

        destination_path = os.path.join(
            FITNESS_ORIGINAL_DIR,
            f"{clean_name}_fitness_of_successful_individuals.png"
        )

        if os.path.exists(source_path):
            shutil.copy2(
                source_path,
                destination_path
            )

            print(
                "Copied:",
                destination_path
            )

        else:
            print(
                "WARNING: Missing fitness plot for",
                obj_name
            )


def write_summary_csv():
    csv_path = os.path.join(
        OUT_DIR,
        "summary_metrics.csv"
    )

    with open(csv_path, "w") as f:
        f.write(
            "object,run,evaluations,"
            "success_archive,"
            "success_rate_percent,"
            "mean_grasp,max_grasp,"
            "fitness_grasp_correlation\n"
        )

        for obj_name, run_path in RUNS.items():
            data = load_data(
                run_path
            )

            archive = load_final_archive(
                run_path
            )

            infos = archive["infos"]

            fitnesses = np.array(
                archive["fitnesses"],
                dtype=float
            )

            grasp_scores = np.array(
                [
                    x["grasp_proxy"]
                    for x in infos
                ],
                dtype=float
            )

            evaluations = int(
                np.array(
                    data["n_evals_hist"]
                )[-1]
            )

            success_archive = len(
                infos
            )

            success_rate = (
                success_archive /
                evaluations
            ) * 100.0

            mean_grasp = float(
                np.mean(
                    grasp_scores
                )
            )

            max_grasp = float(
                np.max(
                    grasp_scores
                )
            )

            if len(fitnesses) > 1:
                corr = float(
                    np.corrcoef(
                        fitnesses,
                        grasp_scores
                    )[0, 1]
                )
            else:
                corr = float("nan")

            run_id = os.path.basename(
                run_path
            )

            f.write(
                f"{obj_name},"
                f"{run_id},"
                f"{evaluations},"
                f"{success_archive},"
                f"{success_rate:.4f},"
                f"{mean_grasp:.6f},"
                f"{max_grasp:.6f},"
                f"{corr:.6f}\n"
            )

    print(
        "Saved:",
        csv_path
    )


save_line_plot(
    filename="combined_qd_score.png",
    title="QD Score Evolution Across Objects",
    ylabel="QD score",
    key="success_archive_qd_score_hist",
)

save_line_plot(
    filename="combined_successful_cells.png",
    title="Successful Archive Growth Across Objects",
    ylabel="Number of successful cells",
    key="n_successful_cells_list",
)

save_line_plot(
    filename="combined_success_coverage.png",
    title="Success Archive Coverage Across Objects",
    ylabel="Success archive coverage",
    key="success_archive_cvg_hist",
)

save_line_plot(
    filename="combined_success_ratio.png",
    title="Success Ratio Evolution Across Objects",
    ylabel="Success ratio (%)",
    key="success_ratio_hist",
    multiply_100=True,
)

plot_final_grasp_friendliness_boxplot()

plot_grasp_friendliness_histogram()

plot_fitness_vs_grasp_scatter()

copy_original_fitness_plots()

write_summary_csv()

print(
    "\nAll outputs saved in:",
    OUT_DIR
)

print(
    "Original per-object fitness plots saved in:",
    FITNESS_ORIGINAL_DIR
)
