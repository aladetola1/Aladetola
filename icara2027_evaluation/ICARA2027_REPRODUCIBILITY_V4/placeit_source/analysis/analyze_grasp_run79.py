import numpy as np

archive = np.load(
    "runs/run79/success_archives/individuals_0.npz",
    allow_pickle=True
)

infos = archive["infos"]
fitnesses = archive["fitnesses"]

grasp_scores = np.array(
    [x["grasp_proxy"] for x in infos]
)

corr = np.corrcoef(
    fitnesses,
    grasp_scores
)[0,1]

print("solutions:", len(infos))
print("fitness-grasp correlation:", corr)

print("max grasp:", np.max(grasp_scores))
print("min grasp:", np.min(grasp_scores))
print("mean grasp:", np.mean(grasp_scores))
