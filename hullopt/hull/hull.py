"""
Hull object, generated from hull parameters.
Including mesh and all info required for simulation
"""

from trimesh import Trimesh
import trimesh
from .params import Params
from .generation import generate_simple_hull, apply_rocker_to_hull, add_cockpit_to_hull
from typing import Optional
from .constraints import Constraints
import numpy as np

class Hull:
  """
  Class for hull objects, generated from a set of parameters, or directly from a mesh
  """
  def __init__(self, params: Optional[Params], from_mesh: Optional[Trimesh] = None) -> None:
    """
    params: Generate hull from params (density, etc.)
    from_mesh: Generate from specified trimesh instead
    """
    # Set unmodified params
    self.density: float = params.density
    self.hull_thickness: float = params.hull_thickness
    self.params: Params = params
    
    if from_mesh is None:
      self.mesh: Trimesh = Hull.generate_mesh(params)
    else:
      self.mesh = from_mesh
      self.mesh.density = params.density # Override mesh density with params density

    if not self.mesh.is_watertight:
      # We must have a watertight hull mesh
      raise RuntimeError("Generated/Provided Hull contains Holes")
    
    # Check constraints
    Constraints().check_hull(self)
    
  @classmethod
  def from_mesh(cls, mesh: Trimesh):
    return cls(None, from_mesh=mesh)
        
  @staticmethod
  def generate_mesh(params: Params) -> Trimesh:
    # Generate outer hull mesh
    outer_mesh = generate_simple_hull(
      length=params.length,
      beam=params.beam,
      depth=params.depth,
      cross_section_exponent=params.cross_section_exponent,
      beam_position=params.beam_position
    )

    # Generate inner hull mesh
    inner_mesh = generate_simple_hull(
      length=params.length - 2 * params.hull_thickness,
      beam=params.beam - 2 * params.hull_thickness,
      depth=params.depth - 2 * params.hull_thickness,
      cross_section_exponent=params.cross_section_exponent,
      beam_position=params.beam_position
    )

    # Create a hollow hull shell by subtracting inner from outer
    try:
        mesh = outer_mesh.difference(inner_mesh, engine="manifold")
    except Exception:
        mesh = outer_mesh.difference(inner_mesh)

    # Add cockpit opening
    if params.cockpit_opening:
      mesh = add_cockpit_to_hull(
        mesh,
        length=params.length,
        cockpit_length=params.cockpit_length,
        cockpit_width=params.cockpit_width,
        cockpit_position=params.cockpit_position
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
    mesh.apply_translation([-centroid[0], -centroid[1], 0.0])

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
