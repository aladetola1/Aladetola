import numpy as np

archive = np.load(
    "runs/run76/success_archives/individuals_0.npz",
    allow_pickle=True
)

bd = archive["behavior_descriptors"]

names = [
    "x",
    "y",
    "z",
    "roll",
    "pitch",
    "yaw"
]

for i,n in enumerate(names):
    print(
        n,
        "min:",
        bd[:,i].min(),
        "max:",
        bd[:,i].max(),
        "range:",
        bd[:,i].max()-bd[:,i].min()
    )
