import numpy as np

archive = np.load(
    "runs/run76/success_archives/individuals_0.npz",
    allow_pickle=True
)

infos = archive["infos"]

roll = []
pitch = []
yaw = []

for x in infos:
    rpy = x["final_6dof_pose"]["euler_rpy"]
    roll.append(rpy[0])
    pitch.append(rpy[1])
    yaw.append(rpy[2])


print("roll mean/std:", np.mean(roll), np.std(roll))
print("pitch mean/std:", np.mean(pitch), np.std(pitch))
print("yaw mean/std:", np.mean(yaw), np.std(yaw))


print("\nMost common orientation examples:")

for i in range(10):
    print(infos[i]["final_6dof_pose"]["euler_rpy"])
