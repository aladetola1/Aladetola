
import numpy as np

archive = np.load(
    "runs/run103/success_archives/individuals_0.npz",
    allow_pickle=True
)

infos = archive["infos"]
fitnesses = archive["fitnesses"]

grasp_scores = np.array([x["grasp_proxy"] for x in infos])

print("solutions:", len(infos))
print("fitness-grasp correlation:", np.corrcoef(fitnesses, grasp_scores)[0,1])
print("max grasp:", grasp_scores.max())
print("min grasp:", grasp_scores.min())
print("mean grasp:", grasp_scores.mean())

best_grasp = np.argmax(grasp_scores)
print("\nBest grasp:")
print(infos[best_grasp])

best_fit = np.argmax(fitnesses)
print("\nBest fitness:")
print(infos[best_fit])
print("fitness:", fitnesses[best_fit])
print("grasp:", grasp_scores[best_fit])
