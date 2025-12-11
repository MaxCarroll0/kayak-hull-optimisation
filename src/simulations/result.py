from dataclasses import dataclass

@dataclass
class Result:
    """
    float righting_moment: angular force exerted to right the hull
    float draugh_proportion: [0,1] proportion of hull underwater when floating
    """
    righting_moment: float
    draught_proportion: float
