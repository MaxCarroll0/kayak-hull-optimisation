"""
Analytic Simulation
"""

from functools import reduce, partial
from typing import Tuple, cast
import numpy as np
from scipy import optimize
import trimesh
from trimesh import Trimesh, Scene
from hullopt import config, Hull
from .params import Params
from .result import Result
from .storage import ResultStorage




storage = ResultStorage()

def _iterate_draught(mesh: Trimesh) -> Tuple[int, float]:
  """
  Iterate various water levels (draught) and calculate displacement.
  Returns the draught iterating until displacement = weight
  """
  def required_buoyancy(draught: float):
    _, displacement = _calculate_centre_buoyancy_and_displacement(mesh, draught)
    return mesh.mass - displacement

  lower = mesh.bounds[0][2] + 0.001 # 1mm buffer. TODO: switch to be in terms of draught_threshold
  upper = mesh.bounds[1][2] - 0.001
  draught, draught_result = optimize.bisect(required_buoyancy,
                                  upper,
                                  lower,
                                  # TODO, parameterise draught_threshold based on hull?
                                  xtol=config.hyperparameters.draught_threshold * (upper-lower+0.002),
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

def compose(f, g):
    return lambda *a, **kw: f(g(*a, **kw))

def _reserve_buoyancy(mesh: Trimesh, draught):
  lower = draught + 0.001 # 1mm buffer
  upper = mesh.bounds[1][2] - 0.001
  result = cast(optimize.OptimizeResult,
                optimize.minimize_scalar(compose(
                  lambda t: -t[1],
                  partial(_calculate_centre_buoyancy_and_displacement, mesh)),
                  bounds=(lower, upper),
                  # TODO, parameterise draught_threshold based on hull?
                  options= {
                    'maxiter': config.hyperparameters.draught_max_iterations,
                    'xatol': config.hyperparameters.draught_threshold * (upper-lower+0.002),
                  }))
  return result.nit, -result.fun - mesh.mass

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
  iterations_draught, draught = _iterate_draught(mesh)
  iterations_reserve_buoyancy, reserve_buoyancy = _reserve_buoyancy(mesh, draught)
  new_result = Result(
        righting_moment=_calculate_righting_moment(mesh, draught),
        reserve_buoyancy=reserve_buoyancy,
        scene=_scene_draught(mesh, draught),
        cost=config.hyperparameters.cost_analytic(iterations_draught + iterations_reserve_buoyancy)
    )
  if use_cache:
        storage.store(new_result, params)
  return new_result


__all__ = [ "run" ]
