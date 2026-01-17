import numpy as np
from .hull import Hull
from .params import Params
from .constraints import Constraints


def generate_random_hulls(n: int, cockpit_opening: bool = False, seed: int = 42) -> list[Params]:
    """
    Generate n random hulls that satisfy default constraints. 
    """
    np.random.seed(seed)
    
    # uses default constraints for now, we can define our own as well
    constraints = Constraints()  
    hulls = []
    
    for _ in range(n):

        while True:

            # Get random values from constraint ranges
            length = np.random.uniform(*constraints.length_range)
            beam = np.random.uniform(*constraints.beam_range)
            depth = np.random.uniform(*constraints.depth_range)
            cross_section_exponent = np.random.uniform(*constraints.cross_section_exponent_range)
            beam_position = np.random.uniform(*constraints.beam_position_range)
            rocker_position = 0.5  # Keep rocker position centered
            rocker_exponent = np.random.uniform(*constraints.rocker_exponent_range)
            rocker_bow = np.random.uniform(*constraints.rocker_bow_range)
            rocker_stern = np.random.uniform(max(rocker_bow - 0.05, constraints.rocker_stern_range[0]),
                                            min(rocker_bow, constraints.rocker_stern_range[1]))
            hull_thickness = np.random.uniform(*constraints.hull_thickness_range)
            
            # Check if ratios are satisfied
            length_to_beam = length / beam
            beam_to_depth = beam / depth
            
            valid = (
                constraints.length_to_beam_ratio_range[0] <= length_to_beam <= constraints.length_to_beam_ratio_range[1]
                and constraints.beam_to_depth_ratio_range[0] <= beam_to_depth <= constraints.beam_to_depth_ratio_range[1]
                and rocker_stern <= rocker_bow
            )
            
            if valid:
                # Create hull params
                params = Params(
                    density=900.0,
                    hull_thickness=hull_thickness,
                    length=length,
                    beam=beam,
                    depth=depth,
                    cross_section_exponent=cross_section_exponent,
                    beam_position=beam_position,
                    rocker_bow=rocker_bow,
                    rocker_stern=rocker_stern,
                    rocker_position=rocker_position,
                    rocker_exponent=rocker_exponent,
                    cockpit_opening=cockpit_opening,
                    cockpit_length=0.8,
                    cockpit_width=0.6,
                    cockpit_position=0.5
                )
                try:
                    hull = Hull(params)
                except:
                    continue  # Invalid hull, try again

                hulls.append(hull)
                break
    
    return hulls

