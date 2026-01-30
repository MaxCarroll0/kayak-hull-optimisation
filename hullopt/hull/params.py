from dataclasses import dataclass

@dataclass
class Params:
    """
    Parameters defining a hull.
    Used to create a Hull object.

    density - kg/m^3 (float): density of hull material
    hull_thickness - m (float): thickness of hull material

    length - m (float): overall hull length
    beam - m (float): max width of hull
    depth - m (float): max depth of hull
    
    cross_section_exponent - float: super-ellipse exponent controlling cross-section shape
    beam_position - float: position of maximum beam along hull length (0.0=bow, 1.0=stern)
    
    rocker_bow - m (float): keel curvature at bow
    rocker_stern - m (float): keel curvature at stern
    rocker_position - float: position of minimum rocker along hull length (0.0=bow, 1.0=stern)
    rocker_exponent - float: exponent controlling how banana-shaped the kayak is at each end

    cockpit_opening - bool: whether to include a cockpit opening
    cockpit_length - m (float): length of cockpit opening
    cockpit_width - m (float): width of cockpit opening
    cockpit_position - float: position of cockpit center along hull length ~ at centre
    """

    # Physical properties
    density: float
    hull_thickness: float

    # global dimensions
    length: float
    beam: float
    depth: float

    # cross-section shape
    cross_section_exponent: float
    beam_position: float

    # longitudinal profile
    rocker_bow: float
    rocker_stern: float
    rocker_position: float
    rocker_exponent: float
    
    # cockpit opening
    cockpit_length: float = 0.8
    cockpit_width: float = 0.6
    cockpit_position: float = 0.5
    cockpit_opening: bool = False

    # A better parameterisation that does not violate ratio constraints
    @classmethod
    def from_ratio_parameterisation(cls, density, hull_thickness, length, length_beam_ratio, beam_depth_ratio, cross_section_exponent, beam_position, rocker_bow, rocker_stern, rocker_position, rocker_exponent, cockpit_length_ratio, cockpit_width_ratio, cockpit_position, cockpit_opening):
        return cls(density=density,
                   hull_thickness=hull_thickness,
                   length=length,
                   beam=length/length_beam_ratio,
                   depth=(length/length_beam_ratio)/beam_depth_ratio,
                   cross_section_exponent=cross_section_exponent,
                   beam_position=beam_position,
                   rocker_bow=rocker_bow,
                   rocker_stern=rocker_stern,
                   rocker_position=rocker_position,
                   rocker_exponent=rocker_exponent,
                   cockpit_length=cockpit_length_ratio*length,
                   cockpit_width=cockpit_width_ratio*length/length_beam_ratio,
                   cockpit_position=cockpit_position,
                   cockpit_opening=cockpit_opening)
