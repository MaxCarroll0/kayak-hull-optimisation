from dataclasses import dataclass

@dataclass
class Params:
    """
    Parameters defining a hull.
    Used to create a Hull object.

    density - kg/m^3 (float): 
    """
    density: float
