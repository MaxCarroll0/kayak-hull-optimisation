"""
Hull object, generated from hull parameters.
Including mesh and all info required for simulation
"""

from trimesh import Trimesh
import trimesh
from .params import Params
from typing import Optional

class Hull:
  """
  Class for hull objects, generated from a set of parameters, or directly from a mesh
  """
  def __init__(self, params: Optional[Params], from_mesh: Optional[Trimesh] = None) -> None:
    """
    params: Generate hull from params (density, etc.)
    from_mesh: Generate from specified trimesh instead
    """
    self.params = params
    match (params, from_mesh):
      case None, None:
        raise ValueError("Must specify hull params or mesh")
      case None, mesh:
        self.density = mesh.density
        self.mesh = from_mesh   
      case params, None:
        # Set unmodified params
        self.density: float = params.density
        self.mesh: Trimesh = Hull.generate_mesh(params)    

    if not self.mesh.is_watertight:
      # We must have a watertight hull mesh
      raise RuntimeError("Generated/Provided Hull contains Holes")

    # Set derived params
    self.mass = self.mesh.mass
    
  @classmethod
  def from_mesh(cls, mesh: Trimesh):
    return cls(None, from_mesh=mesh)
        
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
