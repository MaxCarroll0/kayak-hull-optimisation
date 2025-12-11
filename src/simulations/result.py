from dataclasses import dataclass
from trimesh import Scene

@dataclass
class Result:
    """
    righting_moment - Nm (float): angular force exerted to right the hull
    draugh_proportion - % (float in [0,1]): proportion of hull above the waterline when floating
    scene - Trimesh.Scene: scene containing the tilted hull & waterline for viewing with scene.show()
    cost - float: Simulation cost (accounting for # of iterations, and discretisation). Note: does not account for (hardware-dependent) time taken to complete
    """
    righting_moment: float
    draught_proportion: float
    scene: Scene
    cost: float
