import numpy as np

archive = np.load(
    "runs/run76/success_archives/individuals_0.npz",
    allow_pickle=True
)

infos = archive["infos"]
fitness = archive["fitnesses"]

grasp = np.array([
    x["grasp_proxy"]
    for x in infos
])


fitness_rank = np.argsort(fitness)[::-1]
grasp_rank = np.argsort(grasp)[::-1]


print("Top 10 by fitness")
for i in fitness_rank[:10]:
    print(
        i,
        "fitness:",
        fitness[i],
        "grasp:",
        grasp[i]
    )


print("\n\nTop 10 by grasp")
for i in grasp_rank[:10]:
    print(
        i,
        "fitness:",
        fitness[i],
        "grasp:",
        grasp[i]
    )

