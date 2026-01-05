"""
Analytic Simulation
"""

from functools import reduce
from typing import Tuple, Any, cast
import numpy as np
from scipy import optimize
import trimesh
from trimesh import Trimesh, Scene
from hullopt import config
from hullopt.hull import Hull
from hullopt.simulations.params import Params
from hullopt.simulations.result import Result
from hullopt.simulations.storage import ResultStorage




storage = ResultStorage()

def _iterate_draught(mesh: Trimesh) -> Tuple[int, float]:
  """
  Iterate various water levels (draught) and calculate displacement.
  Returns the draught iterating until displacement = weight
  """
  def required_buoyancy(draught: float):
    _, displacement = _calculate_centre_buoyancy_and_displacement(mesh, draught)
    return mesh.mass - displacement

  lower = mesh.bounds[0][2] + 0.001 # 1mm buffer
  upper = mesh.bounds[1][2] - 0.001
  draught, draught_result = optimize.bisect(required_buoyancy,
                                  upper,
                                  lower,
                                  # TODO, parameterise draught_threshold based on hull?
                                  xtol=config.hyperparameters.draught_threshold * (upper-lower),
                                  maxiter=config.hyperparameters.draught_max_iterations,
                                  disp=True,
                                  full_output=True)
  return draught_result.iterations, draught

def _calculate_centre_buoyancy_and_displacement(mesh: Trimesh, draught: float) -> Tuple[Tuple[float, float, float], float]:
  """
  Calculate the centre of buoyancy for a given draught level.
  i.e. The centre of mass of the water displaced by the submerged portion and its air pockets.
  """
  submerged = trimesh.intersections.slice_mesh_plane(mesh, [0,0,-1], [0,0,draught], cap=True)
  water_box = trimesh.creation.box(bounds=[submerged.bounds[0] * 1.1, submerged.bounds[1] * [1.1,1.1,1]])
  # Calculate water/air meshes around the boat
  water_diff: Trimesh = trimesh.boolean.difference([water_box, mesh])
  pockets = water_diff.split()  # Get all pockets
  # Exactly ONE pocket corresponds to water, and it is the only pocket to contain points outside the submerged points
  air_pockets = [pocket for pocket in pockets if not pocket.contains([submerged.bounds[0]*1.05])[0]]
  water_displaced = air_pockets + [submerged]
  # Note, all densities reset to 1 by previous operations
  return tuple(trimesh.Scene(water_displaced).center_mass),\
    reduce(lambda acc, m: m.volume + acc, water_displaced, 0) * config.constants.water_density

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
  water_box = trimesh.creation.box(bounds=[submerged.bounds[0] * 1.1, submerged.bounds[1] * [1.1,1.1,0]])
  water_diff: Trimesh = trimesh.boolean.difference([water_box, mesh])
  pockets = water_diff.split()  # Get all pockets
  # Exactly ONE pocket corresponds to water, and it is the only pocket to contain points outside the submerged points
  water = next(pocket for pocket in pockets if pocket.contains([submerged.bounds[0]*1.05])[0])

  water._visual.face_colors = [0,255,240,90]
  mesh._visual.face_colors = [255,0,0,255]
  return trimesh.Scene([mesh, water])

def run(hull: Hull, params: Params, use_cache: bool = True) -> Result:
  T = trimesh.transformations.translation_matrix(hull.mesh.center_mass)
  mesh = hull.mesh.copy().apply_transform(T)
  R = trimesh.transformations.rotation_matrix(params.heel, [1,0,0], hull.mesh.center_mass)
  mesh.apply_transform(R)
  iterations, draught = _iterate_draught(mesh)
  new_result = Result(
        righting_moment=_calculate_righting_moment(mesh, draught),
        draught_proportion=_draught_proportion(mesh, draught),
        scene=_scene_draught(mesh, draught),
        cost=config.hyperparameters.cost_analytic(iterations)
    )
  if use_cache:
        storage.store(new_result, params)
  return new_result


__all__ = [ "run" ]
