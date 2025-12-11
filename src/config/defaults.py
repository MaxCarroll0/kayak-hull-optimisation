"""
Default values for user inputs. e.g. hull constraints
"""
from ..hull import Hull
import trimesh

kayak_density = 1500 # kg/m^3

# A rectangular bathtub-shaped hull
def _bathtub():
    outer_box = trimesh.creation.box(extents=[2.5,0.75,0.5])
    inner_box = trimesh.creation.box(extents=[2.3, 0.55, 0.4])
    inner_box.apply_translation([0, 0, 0.1])
    tub = outer_box.difference(inner_box)
    tub.density = kayak_density
    return tub

hull_bathtub = Hull.from_mesh(_bathtub())
    
