"""
Hull constraint logic
"""

class Constraints:
  def __init__(self,
               # Absolute parameter bounds
               length_range: tuple[float, float]=(1.5,3.0),
               beam_range: tuple[float, float]=(0.6,0.8),
               depth_range: tuple[float, float]=(0.25,0.40),
               hull_thickness_range: tuple[float, float]=(0.002,0.006),
               cross_section_exponent_range: tuple[float, float]=(1.5,3.0),
               beam_position_range: tuple[float, float] = (0.35, 0.65),
               rocker_bow_range: tuple[float, float]=(0.05,0.35) ,
               rocker_stern_range: tuple[float, float]=(0.05,0.35),
               rocker_position_range: tuple[float, float]=(0.35,0.60),
               rocker_exponent_range: tuple[float, float]=(2.0,3.0),

               # Ratio parameter bounds
               length_to_beam_ratio_range: tuple[float, float] = (3.0, 6.5),
               beam_to_depth_ratio_range: tuple[float, float] = (1.5, 2.5),
               ) -> None:
    
    """
    Initialize hull constraints
    Currently these are just placeholders; I have just guessed these based on typical kayak dimensions available online
    TODO: set realistic bounds 
    """
    
    self.length_range = length_range
    self.beam_range = beam_range
    self.depth_range = depth_range
    self.hull_thickness_range = hull_thickness_range
    self.cross_section_exponent_range = cross_section_exponent_range
    self.beam_position_range = beam_position_range
    self.rocker_bow_range = rocker_bow_range
    self.rocker_stern_range = rocker_stern_range
    self.rocker_position_range = rocker_position_range
    self.rocker_exponent_range = rocker_exponent_range
    self.length_to_beam_ratio_range = length_to_beam_ratio_range
    self.beam_to_depth_ratio_range = beam_to_depth_ratio_range

  def check_hull(self, hull):
    # Check hull satisfies constraints
    params = hull.params

    # Check absolute bounds
    absolute_bounds = [
      (self.length_range[0] <= params.length <= self.length_range[1], 
       f'Length {params.length} out of range {self.length_range}'),

      (self.beam_range[0] <= params.beam <= self.beam_range[1], 
       f'Beam {params.beam} out of range {self.beam_range}'),

      (self.depth_range[0] <= params.depth <= self.depth_range[1], 
       f'Depth {params.depth} out of range {self.depth_range}'),

      (self.hull_thickness_range[0] <= params.hull_thickness <= self.hull_thickness_range[1], 
       f'Hull thickness {params.hull_thickness} out of range {self.hull_thickness_range}'),
      
      (self.cross_section_exponent_range[0] <= params.cross_section_exponent <= self.cross_section_exponent_range[1], 
       f'Cross-section exponent {params.cross_section_exponent} out of range {self.cross_section_exponent_range}'),

      (self.beam_position_range[0] <= params.beam_position <= self.beam_position_range[1], 
       f'Beam position {params.beam_position} out of range {self.beam_position_range}'),
       
      (self.rocker_bow_range[0] <= params.rocker_bow <= self.rocker_bow_range[1], 
       f'Rocker bow {params.rocker_bow} out of range {self.rocker_bow_range}'),

      (self.rocker_stern_range[0] <= params.rocker_stern <= self.rocker_stern_range[1], 
       f'Rocker stern {params.rocker_stern} out of range {self.rocker_stern_range}'),

      (self.rocker_position_range[0] <= params.rocker_position <= self.rocker_position_range[1], 
       f'Rocker position {params.rocker_position} out of range {self.rocker_position_range}'),

      (self.rocker_exponent_range[0] <= params.rocker_exponent <= self.rocker_exponent_range[1], 
       f'Rocker exponent {params.rocker_exponent} out of range {self.rocker_exponent_range}'),
    ]

    for check, err_msg in absolute_bounds:
      if not check:
        raise ValueError(f'Hull constraint violation: {err_msg}')
      
    # Check ratio constraints
    length_to_beam = params.length / params.beam
    beam_to_depth = params.beam / params.depth
    
    ratio_checks = [
      (self.length_to_beam_ratio_range[0] <= length_to_beam <= self.length_to_beam_ratio_range[1],
       f'Length/beam ratio {length_to_beam:.2f} out of range {self.length_to_beam_ratio_range}.\nHull is too {"stubby" if length_to_beam < self.length_to_beam_ratio_range[0] else "sleek"}.'),
      
      (self.beam_to_depth_ratio_range[0] <= beam_to_depth <= self.beam_to_depth_ratio_range[1],
       f'Beam/depth ratio {beam_to_depth:.2f} out of range {self.beam_to_depth_ratio_range}.\nHull is too {"deep and narrow" if beam_to_depth < self.beam_to_depth_ratio_range[0] else "wide and shallow"}.'),
      
      (params.rocker_stern <= params.rocker_bow,
       f'Rocker at stern ({params.rocker_stern:.3f}m) should not exceed rocker at bow ({params.rocker_bow:.3f}m).'),

       (abs(params.rocker_bow - params.rocker_stern) <= 0.05,
        f'Rocker bow and stern difference {abs(params.rocker_bow - params.rocker_stern):.3f}m exceeds allowed tolerance of 0.05m.')
    ]
    
    for check, err_msg in ratio_checks:
      if not check:
        raise ValueError(f'Hull constraint violation: {err_msg}')

      
    return True
