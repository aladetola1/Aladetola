import trimesh
import os
import numpy as np

print(os.getcwd())

# cube = trimesh.creation.box(extents=(0.05, 0.05, 0.05))
# sphere = trimesh.creation.icosphere(radius=0.05, subdivisions=3)
# cylinder = trimesh.creation.cylinder(radius=0.03, height=0.1, sections=32)
# cone = trimesh.creation.cone(radius=0.03, height=0.1, sections=32)
# capsule = trimesh.creation.capsule(radius=0.02, height=0.08, count=[16, 16])


# sphere.export('sphere.obj')
# cylinder.export('cylinder.obj')
# cone.export('cone.obj')
# capsule.export('capsule.obj')



width=0.2
depth=0.2
height=0.03


table = trimesh.creation.box(extents=[width, depth, height])
    
table.apply_translation([0, 0, height/2])
    





table.export('table.obj')