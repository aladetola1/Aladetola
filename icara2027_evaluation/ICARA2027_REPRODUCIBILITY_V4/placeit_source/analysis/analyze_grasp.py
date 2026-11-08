import numpy as np

archive = np.load(
    "runs/run76/success_archives/individuals_0.npz",
    allow_pickle=True
)

infos = archive["infos"]
fitnesses = archive["fitnesses"]

grasp_scores = np.array([x["grasp_proxy"] for x in infos])

print("Number of solutions:", len(infos))

print("\nTop 10 grasp proxy solutions:")

top10 = np.argsort(grasp_scores)[-10:][::-1]

for rank, idx in enumerate(top10):
    print("\nRank:", rank+1)
    print("index:", idx)
    print("grasp_proxy:", grasp_scores[idx])
    print("fitness:", fitnesses[idx])
    print("final xyz:", infos[idx]["final_6dof_pose"]["xyz"])
    print("final rpy:", infos[idx]["final_6dof_pose"]["euler_rpy"])
