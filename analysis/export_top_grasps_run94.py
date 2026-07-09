import numpy as np

archive = np.load(
    "runs/run94/success_archives/individuals_0.npz",
    allow_pickle=True
)

infos = archive["infos"]
fitnesses = archive["fitnesses"]

grasp_scores = np.array(
    [x["grasp_proxy"] for x in infos]
)

# Highest grasp first
order = np.argsort(grasp_scores)[::-1]

TOP_N = 10

print(f"Top {TOP_N} graspable placements\n")

for rank, idx in enumerate(order[:TOP_N], start=1):

    print("=" * 60)
    print("Rank:", rank)
    print("Grasp score:", grasp_scores[idx])
    print("Fitness:", fitnesses[idx])
    print("Final pose:")
    print(infos[idx]["final_6dof_pose"])

top_infos = []

for idx in order[:TOP_N]:
    top_infos.append({
        "fitness": float(fitnesses[idx]),
        "grasp_proxy": float(grasp_scores[idx]),
        "final_pose": infos[idx]["final_6dof_pose"]
    })

np.save(
    "runs/run94/top10_grasp_candidates.npy",
    top_infos,
    allow_pickle=True
)

print("\nSaved top 10 grasp candidates.")
