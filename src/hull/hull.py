"""
Hull object, generated from hull parameters.
Including mesh and all info required for simulation
"""

import config
from trimesh import Trimesh
import trimesh
from params import Params
from typing import Tuple, Any, Optional
import numpy as np

def vec3d_to_tuple(vec: np.ndarray[Any, np.dtype[np.float64]]) -> Tuple[float, float, float]:
  return (vec[0], vec[1], vec[2])

class Hull:
  def __init__(self, params: Params, from_mesh: Optional[Trimesh] = None) -> None:
    """
    params: dict: "density" ...
    from_mesh: Generate from specified trimesh instead
    """
    # Set unmodified params
    self.density: float = params.density
    self.heel: float = params.heel
    
    # Generate Mesh
    if from_mesh is None:
      self.mesh: Trimesh = Hull.generate_mesh(params)
    else:
      self.mesh = from_mesh
      self.mesh.density = params.density # Override mesh density with params density
    if self.mesh.is_watertight:
      # We must have a watertight hull mesh
      raise RuntimeError("Generated/Provided Hull contains Holes")

    # Calculate mesh properties
    self.recalculate_properties()

  @classmethod
  def from_mesh(cls, mesh: Trimesh):
    return cls(Params(density=mesh.density, heel=0), from_mesh=mesh)
    
  def recalculate_properties(self) -> None:
    """
    Recalculate properties derived from the mesh (bounds, weight, centre of mass, draught, etc.)
    """
    self.width: float = self.mesh.bounds[0]
    self.length: float = self.mesh.bounds[1]
    self.height: float = self.mesh.bounds[2]

    self.mass: float = self.mesh.mass

    self.centre_of_mass: Tuple[float, float, float] = vec3d_to_tuple(self.mesh.center_mass)
    moments_of_inertia = self.mesh.mass_properties.inertia
    self.i_xx = vec3d_to_tuple(moments_of_inertia[0])
    self.i_yy = vec3d_to_tuple(moments_of_inertia[1])
    self.i_zz = vec3d_to_tuple(moments_of_inertia[2])

    # Draught and buoyancy
    self.draught = Hull.iterate_draught(self.mesh)
    self.centre_of_buoyancy = Hull.calculate_centre_of_buoyancy(self.mesh, self.draught)
        
  @staticmethod
  def generate_mesh(params: Params) -> Trimesh:
    return Trimesh() # TODO

  # TODO: move to analytic.py / abstract to allow dynamic draught calculations
  @staticmethod
  def iterate_draught(mesh: Trimesh) -> float:
    """
    Iterate various water levels (draught) and calculate displacement.
    Returns the draught iterating until displacement = weight
    """
    diff = float("inf")
    draught = mesh.center_mass[2]
    while abs(diff) > config.hyperparameters.buoyancy_threshold:
      submerged = trimesh.intersections.slice_mesh_plane(mesh, [0,0,-1], [0,0,draught])
      # TODO: Count air into displacement
      displacement = submerged.volume * config.constants.water_density
      diff = mesh.mass - displacement
      draught += abs(diff) / mesh.mass * (mesh.bounds[2][1 if draught > 0 else 0] - draught)
    return draught

  # TODO: move to analytic.py
  @staticmethod
  def calculate_centre_of_buoyancy(mesh: Trimesh, draught: float) -> Tuple[float, float, float]:
    """
    Calculate the centre of buoyancy for a given draught level.
    i.e. The centre of mass of the submerged portion.
    """
    submerged = trimesh.intersections.slice_mesh_plane(mesh, [0,0,-1], [0,0,draught], cap=True)
    # TODO: Count air into displacement
    return vec3d_to_tuple(submerged.center_mass)
    
  def save_to_stl(self, filepath: str) -> None:
    if self.mesh is None:
      raise ValueError("Mesh not generated.")
    self.mesh.export(filepath)
    
  @classmethod
  def load_from_stl(cls, filepath: str):
    loaded = trimesh.load(filepath)
    if not isinstance(loaded, trimesh.Trimesh):
      raise ValueError("Loaded STL did not contain a valid mesh.")
    return Hull.from_mesh(loaded)

  def show_mesh(self) -> None:
    if self.mesh is None:
      raise ValueError("Mesh not generated")
    trimesh.Scene(self.mesh).show(viewer="gl")

  def show(self):
    if self.mesh is None:
      raise ValueError("Mesh not generated")
    T = np.linalg.inv(trimesh.geometry.plane_transform(origin=[0,0,1], normal=[0,0,1]))
    trimesh.Scene([self.mesh, trimesh.path.path.creation.grid(side=2, transform=T)]).show()
    
