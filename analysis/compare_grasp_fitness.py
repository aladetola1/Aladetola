import numpy as np
import matplotlib.pyplot as plt

archive = np.load(
    "runs/run76/success_archives/individuals_0.npz",
    allow_pickle=True
)

infos = archive["infos"]
fitnesses = archive["fitnesses"]

grasp_scores = np.array(
    [x["grasp_proxy"] for x in infos]
)

plt.scatter(
    fitnesses,
    grasp_scores
)

plt.xlabel("Stability fitness")
plt.ylabel("Grasp proxy")
plt.title("Fitness vs Grasp Proxy")

plt.savefig(
    "runs/run76/fitness_vs_grasp.png"
)

print("saved plot")
