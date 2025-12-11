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

  @classmethod
  def from_mesh(cls, mesh: Trimesh):
    return cls(Params(density=mesh.density), from_mesh=mesh)
        
  @staticmethod
  def generate_mesh(params: Params) -> Trimesh:
    return Trimesh() # TODO
    
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
