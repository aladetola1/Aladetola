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

    # The original paper-run directories were removed during
    # repository cleaning. Their data_export.pkl files were
    # preserved under paper_runs/<object>/.
    if not os.path.exists(data_path):
        object_name = os.path.basename(
            os.path.dirname(run_path)
        )

        data_path = os.path.join(
            "paper_runs",
            object_name,
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
            linewidth=3.2,
            label=obj_name
        )

    # ---------------------------------------------------------
    # Axis labels: large, normal weight
    # ---------------------------------------------------------
    plt.xlabel(
        "Number of evaluations",
        fontsize=22
    )

    plt.ylabel(
        ylabel,
        fontsize=22
    )

    # ---------------------------------------------------------
    # Plot title: large, normal weight
    # ---------------------------------------------------------
    plt.title(
        title,
        fontsize=22,
        pad=12
    )

    # ---------------------------------------------------------
    # Tick labels
    # ---------------------------------------------------------
    plt.xticks(
        fontsize=18
    )

    plt.yticks(
        fontsize=18
    )

    # ---------------------------------------------------------
    # Grid
    # ---------------------------------------------------------
    plt.grid(
        True,
        alpha=0.3,
        linewidth=0.9
    )

    # ---------------------------------------------------------
    # Legend: large and bold object names
    # ---------------------------------------------------------
    legend = plt.legend(
        fontsize=17,
        frameon=True
    )

    for text in legend.get_texts():
        text.set_fontweight("bold")

    plt.tight_layout()

    # ---------------------------------------------------------
    # Save PNG
    # ---------------------------------------------------------
    save_path = os.path.join(
        OUT_DIR,
        filename
    )

    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight"
    )

    # ---------------------------------------------------------
    # Save vector PDF
    # ---------------------------------------------------------
    pdf_path = os.path.splitext(save_path)[0] + ".pdf"

    plt.savefig(
        pdf_path,
        format="pdf",
        bbox_inches="tight"
    )

    plt.close()

    print("Saved:", save_path)
    print("Saved:", pdf_path)

def plot_final_grasp_friendliness_boxplot():
    labels = []
    all_grasp_scores = []

    for obj_name, run_path in RUNS.items():

        # ---------------------------------------------------------
        # Load preserved canonical paper-run data
        # ---------------------------------------------------------
        data = load_data(run_path)

        poses = np.asarray(
            data["final_6dof_pos"],
            dtype=object
        )

        # ---------------------------------------------------------
        # Keep only mechanically stable placements.
        #
        # This exactly reproduces the canonical stable-placement
        # counts:
        # Mug 481, Bowl 217, Spoon 389,
        # Power Drill 549, Plate 242, Banana 213.
        # ---------------------------------------------------------
        stable_poses = [
            p for p in poses
            if float(p["xyz"][2]) > 0
        ]

        # ---------------------------------------------------------
        # Compute GFI using the frozen implementation:
        #
        # H = z
        # D = exp(-d / 0.1)
        # U = exp(-sqrt(roll^2 + pitch^2))
        # GFI = 0.5 H + 0.3 D + 0.2 U
        #
        # Yaw is intentionally unused.
        # ---------------------------------------------------------
        grasp_scores = []

        for pose in stable_poses:
            xyz = np.asarray(
                pose["xyz"],
                dtype=float
            )

            rpy = np.asarray(
                pose["euler_rpy"],
                dtype=float
            )

            roll, pitch, yaw = rpy

            height_score = xyz[2]

            center_distance = np.linalg.norm(
                xyz[:2]
            )

            tilt = np.sqrt(
                roll**2 + pitch**2
            )

            center_score = np.exp(
                -center_distance / 0.1
            )

            upright_score = np.exp(
                -tilt
            )

            grasp_proxy = (
                0.5 * height_score
                + 0.3 * center_score
                + 0.2 * upright_score
            )

            grasp_scores.append(
                grasp_proxy
            )

        grasp_scores = np.asarray(
            grasp_scores,
            dtype=float
        )

        labels.append(obj_name)
        all_grasp_scores.append(
            grasp_scores
        )

        print(
            f"{obj_name}: "
            f"n={len(grasp_scores)}, "
            f"mean={np.mean(grasp_scores):.4f}, "
            f"median={np.median(grasp_scores):.4f}, "
            f"min={np.min(grasp_scores):.4f}, "
            f"max={np.max(grasp_scores):.4f}"
        )

    # ---------------------------------------------------------
    # Figure
    # ---------------------------------------------------------
    plt.figure(figsize=(9, 6))

    bp = plt.boxplot(
        all_grasp_scores,
        labels=labels,
        showmeans=True,
        patch_artist=False,

        boxprops=dict(
            linewidth=2.2
        ),

        whiskerprops=dict(
            linewidth=2.0
        ),

        capprops=dict(
            linewidth=2.0
        ),

        meanprops=dict(
            marker="o",
            markerfacecolor="white",
            markeredgecolor="black",
            markeredgewidth=1.8,
            markersize=9
        ),

        medianprops=dict(
            linewidth=2.4
        ),

        flierprops=dict(
            marker="o",
            markersize=4,
            alpha=0.65
        )
    )

    # ---------------------------------------------------------
    # Legend: Mean and Median
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
            markeredgewidth=1.8,
            markersize=9,
            label="Mean"
        ),
        Line2D(
            [0],
            [0],
            color="tab:orange",
            linewidth=2.4,
            label="Median"
        )
    ]

    legend = plt.legend(
        handles=legend_elements,
        loc="upper right",
        ncol=2,
        fontsize=17,
        frameon=True,
        columnspacing=1.2,
        handletextpad=0.5
    )

    for text in legend.get_texts():
        text.set_fontweight("bold")

    # ---------------------------------------------------------
    # Axis labels
    # Same visual standard as Fig. 4
    # ---------------------------------------------------------
    plt.xlabel(
        "Object",
        fontsize=22
    )

    plt.ylabel(
        "Grasp Friendliness Index (GFI)",
        fontsize=22
    )

    # ---------------------------------------------------------
    # Title
    # ---------------------------------------------------------
    plt.title(
        "Distribution of GFI Across Stable Placements",
        fontsize=22,
        pad=12
    )

    # ---------------------------------------------------------
    # Tick labels
    # ---------------------------------------------------------
    plt.xticks(
        rotation=25,
        fontsize=18
    )

    plt.yticks(
        fontsize=18
    )

    # ---------------------------------------------------------
    # Grid
    # ---------------------------------------------------------
    plt.grid(
        True,
        axis="y",
        alpha=0.3,
        linewidth=0.9
    )

    plt.tight_layout()

    # ---------------------------------------------------------
    # Save high-resolution PNG
    # ---------------------------------------------------------
    save_path = os.path.join(
        OUT_DIR,
        "combined_gfi_boxplot.png"
    )

    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight"
    )

    # ---------------------------------------------------------
    # Save vector PDF
    # ---------------------------------------------------------
    pdf_path = os.path.splitext(
        save_path
    )[0] + ".pdf"

    plt.savefig(
        pdf_path,
        format="pdf",
        bbox_inches="tight"
    )

    plt.close()

    print("Saved:", save_path)
    print("Saved:", pdf_path)

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

        # ---------------------------------------------------------
        # Load preserved canonical paper-run data
        # ---------------------------------------------------------
        data = load_data(run_path)

        poses = np.asarray(
            data["final_6dof_pos"],
            dtype=object
        )

        fitnesses = np.asarray(
            data["fitness_list"],
            dtype=float
        )

        # ---------------------------------------------------------
        # Keep only mechanically stable placements.
        # This reproduces the canonical stable-placement counts.
        # ---------------------------------------------------------
        stable_poses = [
            p for p in poses
            if float(p["xyz"][2]) > 0
        ]

        # ---------------------------------------------------------
        # Safety check:
        # stable poses and preserved fitness values must match.
        # ---------------------------------------------------------
        if len(stable_poses) != len(fitnesses):
            raise ValueError(
                f"{obj_name}: stable poses "
                f"({len(stable_poses)}) != fitness values "
                f"({len(fitnesses)})"
            )

        # ---------------------------------------------------------
        # Compute GFI using the frozen implementation:
        #
        # H = z
        # D = exp(-d / 0.1)
        # U = exp(-sqrt(roll^2 + pitch^2))
        # GFI = 0.5 H + 0.3 D + 0.2 U
        #
        # Yaw is intentionally unused.
        # ---------------------------------------------------------
        grasp_scores = []

        for pose in stable_poses:
            xyz = np.asarray(
                pose["xyz"],
                dtype=float
            )

            rpy = np.asarray(
                pose["euler_rpy"],
                dtype=float
            )

            roll, pitch, yaw = rpy

            height_score = xyz[2]

            center_distance = np.linalg.norm(
                xyz[:2]
            )

            tilt = np.sqrt(
                roll**2 + pitch**2
            )

            center_score = np.exp(
                -center_distance / 0.1
            )

            upright_score = np.exp(
                -tilt
            )

            grasp_proxy = (
                0.5 * height_score
                + 0.3 * center_score
                + 0.2 * upright_score
            )

            grasp_scores.append(
                grasp_proxy
            )

        grasp_scores = np.asarray(
            grasp_scores,
            dtype=float
        )

        # ---------------------------------------------------------
        # Scatter plot
        # ---------------------------------------------------------
        plt.scatter(
            fitnesses,
            grasp_scores,
            s=32,
            alpha=0.60,
            label=obj_name
        )

        print(
            f"{obj_name}: "
            f"n={len(grasp_scores)}, "
            f"fitness=[{np.min(fitnesses):.4f}, "
            f"{np.max(fitnesses):.4f}], "
            f"GFI=[{np.min(grasp_scores):.4f}, "
            f"{np.max(grasp_scores):.4f}]"
        )

    # ---------------------------------------------------------
    # Axis labels
    # Same visual standard as Figs. 4 and 5
    # ---------------------------------------------------------
    plt.xlabel(
        "PlaceIt fitness",
        fontsize=22
    )

    plt.ylabel(
        "Grasp Friendliness Index (GFI)",
        fontsize=22
    )

    # ---------------------------------------------------------
    # Title
    # ---------------------------------------------------------
    plt.title(
        "PlaceIt Fitness vs. GFI",
        fontsize=22,
        pad=12
    )

    # ---------------------------------------------------------
    # Tick labels
    # ---------------------------------------------------------
    plt.xticks(
        fontsize=18
    )

    plt.yticks(
        fontsize=18
    )

    # ---------------------------------------------------------
    # Grid
    # ---------------------------------------------------------
    plt.grid(
        True,
        alpha=0.3,
        linewidth=0.9
    )

    # ---------------------------------------------------------
    # Legend
    # ---------------------------------------------------------
    legend = plt.legend(
        fontsize=17,
        frameon=True
    )

    for text in legend.get_texts():
        text.set_fontweight("bold")

    plt.tight_layout()

    # ---------------------------------------------------------
    # Save high-resolution PNG
    # ---------------------------------------------------------
    save_path = os.path.join(
        OUT_DIR,
        "combined_fitness_vs_grasp_scatter.png"
    )

    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight"
    )

    # ---------------------------------------------------------
    # Save vector PDF
    # ---------------------------------------------------------
    pdf_path = os.path.splitext(
        save_path
    )[0] + ".pdf"

    plt.savefig(
        pdf_path,
        format="pdf",
        bbox_inches="tight"
    )

    plt.close()

    print("Saved:", save_path)
    print("Saved:", pdf_path)

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


#def write_summary_csv():
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

#plot_grasp_friendliness_histogram()

plot_fitness_vs_grasp_scatter()

#copy_original_fitness_plots()

#write_summary_csv()

print(
    "\nAll outputs saved in:",
    OUT_DIR
)

print(
    "Original per-object fitness plots saved in:",
    FITNESS_ORIGINAL_DIR
)
