"""
Analytic Simulation
"""

import config
from hull.hull import Hull
import numpy as np
import trimesh
from trimesh import Trimesh
from typing import Tuple, Any, Optional

def _vec3d_to_tuple(vec: np.ndarray[Any, np.dtype[np.float64]]) -> Tuple[float, float, float]:
  return (vec[0], vec[1], vec[2])

def calculate_centre_of_buoyancy(hull: Hull, draught: float) -> Tuple[float, float, float]:
  """
  Calculate the centre of buoyancy for a given draught level.
  i.e. The centre of mass of the submerged portion.
  """
  submerged = trimesh.intersections.slice_mesh_plane(hull.mesh, [0,0,-1], [0,0,draught], cap=True)
  # TODO: Count air into displacement
  return _vec3d_to_tuple(submerged.center_mass)

def iterate_draught(hull: Hull) -> float:
  """
  Iterate various water levels (draught) and calculate displacement.
  Returns the draught iterating until displacement = weight
  """
  diff = float("inf")
  draught = hull.mesh.center_mass[2]
  while abs(diff) > config.hyperparameters.buoyancy_threshold:
    submerged = trimesh.intersections.slice_mesh_plane(hull.mesh, [0,0,-1], [0,0,draught])
    # TODO: Count air into displacement
    displacement = submerged.volume * config.constants.water_density
    diff = hull.mesh.mass - displacement
    draught += abs(diff) / hull.mesh.mass * (hull.mesh.bounds[2][1 if draught > 0 else 0] - draught)
  return draught

def recalculate_properties(hull: Hull) -> None:
  """
  Recalculate properties derived from the mesh (bounds, weight, centre of mass, draught, etc.)
  """
  mass: float = hull.mesh.mass
  centre_of_mass: Tuple[float, float, float] = _vec3d_to_tuple(hull.mesh.center_mass)
  moments_of_inertia = hull.mesh.mass_properties.inertia
  i_xx = _vec3d_to_tuple(moments_of_inertia[0])
  i_yy = _vec3d_to_tuple(moments_of_inertia[1])
  i_zz = _vec3d_to_tuple(moments_of_inertia[2])
  # Draught and buoyancy
  draught = iterate_draught(hull)
  centre_of_buoyancy = calculate_centre_of_buoyancy(hull, draught)

def show_draught(hull: Hull, draught: float):
  if hull.mesh is None:
    raise ValueError("Mesh not generated")
  T = np.linalg.inv(trimesh.geometry.plane_transform(origin=[0,0,1], normal=[0,0,1]))
  trimesh.Scene([hull.mesh, trimesh.path.path.creation.grid(side=2, transform=T)]).show()

def run(hull, fluid_dir, fluid_flow, angle):
  return {"righting_moment":0.0, "drag":0.0, "cost":0.0}
