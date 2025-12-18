"""
Hull object, generated from hull parameters.
Including mesh and all info required for simulation
"""

from ..config import *
from trimesh import Trimesh
import trimesh
from .params import Params
from .generation import generate_simple_hull, apply_rocker_to_hull
from typing import Optional

class Hull:
  """
  Class for hull objects, generated from a set of parameters, or directly from a mesh
  """
  def __init__(self, params: Params, from_mesh: Optional[Trimesh] = None) -> None:
    """
    params: dict: "density" ...
    from_mesh: Generate from specified trimesh instead
    """
    # Set unmodified params
    self.density: float = params.density
    self.hull_thickness: float = params.hull_thickness
    
    if from_mesh is None:
      self.mesh: Trimesh = Hull.generate_mesh(params)
    else:
      self.mesh = from_mesh
      self.mesh.density = params.density # Override mesh density with params density
    
    # Set derived properties
    self.mass = self.mesh.area * self.hull_thickness * self.density
    # Override mesh density so mesh.mass = volume * density equals our shell mass
    self.mesh.density = self.mass / self.mesh.volume

    # override center of mass to mimic as an empty hollow hull
    min_z = self.mesh.bounds[0][2]
    self.mesh.center_mass = [0.0, 0.0, min_z + (params.depth * 0.25)]

    if not self.mesh.is_watertight:
      # We must have a watertight hull mesh
      raise RuntimeError("Generated/Provided Hull contains Holes")
    
  @classmethod
  def from_mesh(cls, mesh: Trimesh):
    return cls(Params(density=mesh.density), from_mesh=mesh)
        
  @staticmethod
  def generate_mesh(params: Params) -> Trimesh:
    # Generate outer hull - this represents the hull boundary for buoyancy calculations
    mesh = generate_simple_hull(
      length=params.length,
      beam=params.beam,
      depth=params.depth,
      cross_section_exponent=params.cross_section_exponent
    )

    # Apply rocker deformation
    mesh = apply_rocker_to_hull(
      mesh,
      length=params.length,
      rocker_bow=params.rocker_bow,
      rocker_stern=params.rocker_stern,
      rocker_position=params.rocker_position,
      rocker_exponent=params.rocker_exponent
    )

    # Center the mesh
    centroid = mesh.center_mass
    mesh.apply_translation([-centroid[0], 0.0, 0.0])

    return mesh
    
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

  def show(self) -> None:
    if self.mesh is None:
      raise ValueError("Mesh not generated")
    trimesh.Scene(self.mesh).show(viewer="gl")

