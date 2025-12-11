"""
Analytic Simulation
"""

import config
from hull.hull import Hull
import numpy as np
import trimesh
from trimesh import Scene
from typing import Tuple, Any
from simulations.params import Params
from simulations.result import Result

def _vec3d_to_tuple(vec: np.ndarray[Any, np.dtype[np.float64]]) -> Tuple[float, float, float]:
  return (vec[0], vec[1], vec[2])

def _calculate_centre_of_buoyancy(hull: Hull, draught: float) -> Tuple[float, float, float]:
  """
  Calculate the centre of buoyancy for a given draught level.
  i.e. The centre of mass of the submerged portion.
  """
  submerged = trimesh.intersections.slice_mesh_plane(hull.mesh, [0,0,-1], [0,0,draught], cap=True)
  # TODO: Count air into displacement
  return _vec3d_to_tuple(submerged.center_mass)

def _iterate_draught(hull: Hull) -> Tuple[int, float]:
  """
  Iterate various water levels (draught) and calculate displacement.
  Returns the draught iterating until displacement = weight
  """
  diff = float("inf")
  draught = hull.mesh.center_mass[2]
  loops = 0
  while abs(diff) > config.hyperparameters.buoyancy_threshold:
    loops += 1
    if loops > config.hyperparameters.buoyancy_max_iterations:
      raise RuntimeError("Analytic draught calculation failed to converge")
    submerged = trimesh.intersections.slice_mesh_plane(hull.mesh, [0,0,-1], [0,0,draught], cap=True)
    # TODO: Count air into displacement
    displacement = submerged.volume * config.constants.water_density
    diff = hull.mesh.mass - displacement
    draught += abs(diff) / hull.mesh.mass * (hull.mesh.bounds[2][1 if draught > 0 else 0] - draught)
  return loops, draught

def _draught_proportion(hull: Hull, draught: float):
  submerged = trimesh.intersections.slice_mesh_plane(hull.mesh, [0,0,-1], [0,0,draught], cap=True)
  unsubmerged = trimesh.intersections.slice_mesh_plane(hull.mesh, [0,0,1], [0,0,draught], cap=True)
  return unsubmerged.volume / (unsubmerged.volume + submerged.volume)

def _scene_draught(hull: Hull, draught: float) -> Scene:
  T = np.linalg.inv(trimesh.geometry.plane_transform(origin=[0,0,1], normal=[0,0,1]))
  return trimesh.Scene([hull.mesh, trimesh.path.path.creation.grid(side=2, transform=T)])

def run(hull: Hull, params: Params) -> Result:
  iterations, draught = _iterate_draught(hull)
  return Result(righting_moment = 0, # TODO
                draught_proportion = _draught_proportion(hull, draught),
                scene = _scene_draught(hull, draught),
                cost = hyperparameters.cost_analytic(iterations))
