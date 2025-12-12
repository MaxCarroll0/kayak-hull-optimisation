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
    _, displacement = _calculate_centre_buoyancy_and_displacement(mesh, draught)
    fake_displacement = 2 * displacement
    diff = mesh.mass - fake_displacement
    draught += abs(diff) / mesh.mass * (mesh.bounds[1 if diff> 0 else 0][2] - draught)
  return loops, draught

def _calculate_centre_buoyancy_and_displacement(mesh: Trimesh, draught: float) -> Tuple[Tuple[float, float, float], float]:
  """
  Calculate the centre of buoyancy for a given draught level.
  i.e. The centre of mass of the submerged portion.
  """
  submerged = trimesh.intersections.slice_mesh_plane(mesh, [0,0,-1], [0,0,draught], cap=True)
  
  return tuple(submerged.center_mass), submerged.volume * config.constants.water_density

def _calculate_righting_moment(mesh: Trimesh, draught: float) -> Tuple[float, float, float]:
  cob, _ = _calculate_centre_buoyancy_and_displacement(mesh, draught)
  righting_lever = cob - mesh.center_mass
  gravity_force = mesh.mass * config.constants.gravity_on_earth * np.array([0,0,-1])
  righting_moment = np.cross(righting_lever, gravity_force)
  return tuple(righting_moment)

def _draught_proportion(mesh: Trimesh, draught: float):
  submerged = trimesh.intersections.slice_mesh_plane(mesh, [0,0,-1], [0,0,draught], cap=True)
  unsubmerged = trimesh.intersections.slice_mesh_plane(mesh, [0,0,1], [0,0,draught], cap=True)
  return unsubmerged.volume / (unsubmerged.volume + submerged.volume)

def _scene_draught(mesh: Trimesh, draught: float) -> Scene:
  submerged = trimesh.intersections.slice_mesh_plane(mesh, [0,0,-1], [0,0,draught], cap=True)
  water_box = trimesh.creation.box(bounds=[submerged.bounds[0]+0.1, submerged.bounds[1]+[0.1,0.1,0]])
  water_box._visual.face_colors = [0,255,240,50]
  mesh._visual.face_colors = [255,0,0,255]
  return trimesh.Scene([mesh, water_box])

def run(hull: Hull, params: Params) -> Result:
  R = trimesh.transformations.rotation_matrix(params.heel, [1,0,0], hull.mesh.center_mass)
  rotated_mesh = hull.mesh.copy().apply_transform(R)
  iterations, draught = _iterate_draught(rotated_mesh)
  return Result(righting_moment = _calculate_righting_moment(rotated_mesh, draught),
                draught_proportion = _draught_proportion(rotated_mesh, draught),
                scene = _scene_draught(rotated_mesh, draught),
                cost = config.hyperparameters.cost_analytic(iterations))

__all__ = [ "run" ]
