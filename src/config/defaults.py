"""
Default values for user inputs. e.g. hull constraints
"""
from ..hull import Hull, Params
import trimesh

kayak_density = 900 # kg/m^3

# A rectangular bathtub-shaped hull
def _bathtub():
    outer_box = trimesh.creation.box(extents=[2.5,0.75,0.5])
    inner_box = trimesh.creation.box(extents=[2.3, 0.55, 0.4])
    inner_box.apply_translation([0, 0, 0.1])
    tub = outer_box.difference(inner_box)
    tub.density = kayak_density
    return tub

# hull_bathtub = Hull.from_mesh(_bathtub())
    
dummy_hull = Hull(Params(
    density=900.0,
    hull_thickness=0.005,
    length=2.6,
    beam=0.65,
    depth=0.35,
    cross_section_exponent=2.0,
    beam_position=0.50,
    rocker_bow=0.30,
    rocker_stern=0.25,
    rocker_position=0.50,
    rocker_exponent=2.0,
    cockpit_length=0.85,
    cockpit_width=0.50,
    cockpit_position=0.50
))