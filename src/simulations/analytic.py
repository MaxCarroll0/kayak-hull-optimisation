"""
Analytic Simulation
"""

from .. import config
from ..hull import Hull
import numpy as np
import trimesh
from trimesh import Trimesh, Scene
from typing import Tuple, Any, cast
from .params import Params
from .result import Result

def _vec3d_to_tuple(vec: np.ndarray[Any, np.dtype[np.float64]]) -> Tuple[float, float, float]:
  return (vec[0], vec[1], vec[2])

def _iterate_draught(mesh: Trimesh) -> Tuple[int, float]:
  """
  Iterate various water levels (draught) and calculate displacement.
  Returns the draught iterating until displacement = weight
  """
  diff = float("inf")
  draught = mesh.center_mass[2]
  loops = 0
  while abs(diff) > config.hyperparameters.buoyancy_threshold:
    loops += 1
    if loops > config.hyperparameters.buoyancy_max_iterations:
      raise RuntimeError("Analytic draught calculation failed to converge")
    submerged = trimesh.intersections.slice_mesh_plane(mesh, [0,0,-1], [0,0,draught], cap=True)
    # TODO: Count air into displacement. temporarily double displacement to account for this
    displacement = submerged.volume * config.constants.water_density
    fake_displacement = 2 * displacement
    diff = mesh.mass - fake_displacement
    draught += abs(diff) / mesh.mass * (mesh.bounds[1 if diff> 0 else 0][2] - draught)
  return loops, draught

def _calculate_centre_of_buoyancy(mesh: Trimesh, draught: float) -> Tuple[float, float, float]:
  """
  Calculate the centre of buoyancy for a given draught level.
  i.e. The centre of mass of the submerged portion.
  """
  submerged = trimesh.intersections.slice_mesh_plane(mesh, [0,0,-1], [0,0,draught], cap=True)
  # TODO: Count air into displacement
  return submerged.center_mass

def _calculate_righting_moment(mesh: Trimesh, draught: float) -> Tuple[float, float, float]:
  cob = _calculate_centre_of_buoyancy(mesh, draught)
  righting_lever = cob - mesh.center_mass
  gravity_force = mesh.mass * config.constants.gravity_on_earth * np.array([0,0,-1])
  righting_moment = np.cross(righting_lever, gravity_force)
  return cast(Tuple[float, float, float], righting_moment)

def _draught_proportion(mesh: Trimesh, draught: float):
  submerged = trimesh.intersections.slice_mesh_plane(mesh, [0,0,-1], [0,0,draught], cap=True)
  unsubmerged = trimesh.intersections.slice_mesh_plane(mesh, [0,0,1], [0,0,draught], cap=True)
  return unsubmerged.volume / (unsubmerged.volume + submerged.volume)

def _scene_draught(mesh: Trimesh, draught: float) -> Scene:
  T = np.linalg.inv(trimesh.geometry.plane_transform(origin=[0,0,draught], normal=[0,0,1]))
  axes = trimesh.creation.axis(origin_size=0.1, axis_length=0.5)
  return trimesh.Scene([mesh, axes, trimesh.path.path.creation.grid(side=2, transform=T)])

def run(hull: Hull, params: Params) -> Result:
  R = trimesh.transformations.rotation_matrix(params.heel, [1,0,0], hull.mesh.center_mass)
  rotated_mesh = hull.mesh.copy().apply_transform(R)
  iterations, draught = _iterate_draught(rotated_mesh)
  return Result(righting_moment = _calculate_righting_moment(rotated_mesh, draught),
                draught_proportion = _draught_proportion(rotated_mesh, draught),
                scene = _scene_draught(rotated_mesh, draught),
                cost = config.hyperparameters.cost_analytic(iterations))

__all__ = [ "run" ]
