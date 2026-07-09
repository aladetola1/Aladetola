import numpy as np

archive = np.load(
    "runs/run76/success_archives/individuals_0.npz",
    allow_pickle=True
)

infos = archive["infos"]

xyz = np.array([
    x["final_6dof_pose"]["xyz"]
    for x in infos
])

rpy = np.array([
    x["final_6dof_pose"]["euler_rpy"]
    for x in infos
])

grasp = np.array([
    x["grasp_proxy"]
    for x in infos
])


print("solutions:", len(infos))

print("\nFinal height statistics")
print("max height:", np.max(xyz[:,2]))
print("min height:", np.min(xyz[:,2]))
print("mean height:", np.mean(xyz[:,2]))


print("\nGrasp proxy statistics")
print("max:", np.max(grasp))
print("min:", np.min(grasp))
print("mean:", np.mean(grasp))


print("\nCorrelation with height:")
print(np.corrcoef(grasp, xyz[:,2])[0,1])


print("\nTop 5 highest objects:")
idx = np.argsort(xyz[:,2])[-5:][::-1]

for i in idx:
    print("\nindex:", i)
    print("height:", xyz[i,2])
    print("grasp:", grasp[i])
    print("xyz:", xyz[i])
