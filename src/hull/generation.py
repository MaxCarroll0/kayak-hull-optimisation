"""
Mesh generation for kayak hulls based on parameter sets
"""

import numpy as np
from trimesh import Trimesh
from typing import Tuple, Any, cast

def super_ellipse(angle: float, width: float, height: float, n: float) -> Tuple[float, float]:
    """
    Takes an angle in radians, width and height, and exponent n
    Returns a point that defines hull cross-section on the super-ellipse defined by those parameters
    """
    c = np.cos(angle)
    s = np.sin(angle)
    x_raw = np.sign(c) * (np.abs(c) ** (2 / n))
    z_raw = np.sign(s) * (np.abs(s) ** (2 / n))
    return x_raw * width, z_raw * height

def generate_simple_hull(length: float, beam: float, depth: float, cross_section_exponent: float, N_STATIONS: int=60, N_POINTS: int=32) -> Trimesh:
    """
    Generate a simple  mesh (with no rocker) based on global dimensions and cross-section shape of the hull
    """

    vertices = []
    faces = []

    # setting up intial structure
    stat_vals = np.linspace(0, 1.0, N_STATIONS) # normalized points along the hull overall length (easily scale it later)
    angle_vals = np.linspace(0, 2 * np.pi, N_POINTS, endpoint=False)  

    # Generate vertices
    for stat in stat_vals:
        x_pos = length * stat
        width_taper_factor = np.sin(np.pi * stat)  # simple tapering function 0 at bow/stern, 1 at midship

        curr_width = max(beam * width_taper_factor, 1e-4)
        curr_depth = max(depth * width_taper_factor, 1e-4)

        for angle in angle_vals:
            if 0 <= angle <= np.pi:
                y_pos, z_raw = super_ellipse(angle, curr_width / 2.0, curr_depth, cross_section_exponent)
                # bottom half of hull
                z_pos = -np.abs(z_raw)  # lower half is fully submerged
            else:
                # top half of hull
                y_pos, z_raw = super_ellipse(angle, curr_width / 2.0, curr_depth, n=2.0)
                z_pos = np.abs(z_raw) * 0.3  # upper half is shallower to create a deck
            
            vertices.append([x_pos, y_pos, z_pos])

    # Generate faces
    for i in range(N_STATIONS - 1):
        for j in range(N_POINTS):
            # indices of current station
            curr_stat_idx = i * N_POINTS + j
            curr_stat_next_idx = i * N_POINTS + ((j + 1) % N_POINTS)

            # indices of next station
            next_stat_idx = (i + 1) * N_POINTS + j
            next_stat_next_idx = (i + 1) * N_POINTS + ((j + 1) % N_POINTS)
            # two triangles per quad
            faces.append([curr_stat_idx, curr_stat_next_idx, next_stat_next_idx])
            faces.append([curr_stat_idx, next_stat_next_idx, next_stat_idx])

    # Close bow and stern
    bow_tip_idx = len(vertices)
    stern_tip_idx = len(vertices) + 1

    vertices.append([0.0, 0.0, 0.0])  # bow tip
    vertices.append([length, 0.0, 0.0])  # stern tip

    # connect bow and stern faces so it's not open at ends. We need a watertight mesh
    for j in range(N_POINTS):
        next_j = (j + 1) % N_POINTS
        # Bow faces
        faces.append([bow_tip_idx, j, next_j])
        # Stern faces
        last_ring_start = (N_STATIONS - 1) * N_POINTS
        faces.append([stern_tip_idx, last_ring_start + next_j, last_ring_start + j])
    
    hull_mesh = Trimesh(vertices=np.array(vertices), faces=np.array(faces), process=True)
    hull_mesh.fix_normals()

    return hull_mesh

def apply_rocker_to_hull(mesh: Trimesh, length: float, rocker_bow: float, rocker_stern: float, rocker_position: float, rocker_exponent: float) -> Trimesh:
    """
    Deforms a straight hull mesh to apply longitudinal curvature (rocker)
    """
    vertices = mesh.vertices.copy()
    x_coords = (vertices[:, 0] / length)  # normalized x-coordinates [0, 1]
    z_cords = vertices[:, 2]

    z_offsets = np.zeros_like(z_cords)
    is_bow_side = x_coords <= rocker_position
    is_stern_side = x_coords > rocker_position

    # Bow side rocker
    if np.any(is_bow_side):
        x_bow = x_coords[is_bow_side]
        pivot = max(rocker_position, 1e-6)
        rocker_bow_z = rocker_bow * (1 - (x_bow / pivot)) ** rocker_exponent
        z_offsets[is_bow_side] = rocker_bow_z

    # Stern side rocker
    if np.any(is_stern_side):
        x_stern = x_coords[is_stern_side]
        remain_len = max(1.0 - rocker_position, 1e-6)
        rocker_stern_z = rocker_stern * ((x_stern - rocker_position) / remain_len) ** rocker_exponent
        z_offsets[is_stern_side] = rocker_stern_z

    vertices[:, 2] = z_cords + z_offsets
    mesh.vertices = vertices
    mesh.fix_normals()

    return mesh  