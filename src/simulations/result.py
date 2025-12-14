from dataclasses import dataclass, asdict
from trimesh import Scene
from typing import Tuple


@dataclass
class Result:
    """
    righting_moment - Nm (float, float, float): angular forces exerted on the hull by buoyancy & fluid flow (note 3 dimensions x,y,z)
    draugh_proportion - % (float in [0,1]): proportion of hull above the waterline when floating
    scene - Trimesh.Scene: scene containing the tilted hull & waterline for viewing with scene.show()
    cost - float: Simulation cost (accounting for # of iterations, and discretisation). Note: does not account for (hardware-dependent) time taken to complete
    """


    righting_moment: Tuple[float, float, float]
    draught_proportion: float
    scene: Scene
    cost: float

    def righting_moment_heel(self): return self.righting_moment[0]
    def righting_moment_pitch(self): return self.righting_moment[1]
    def righting_moment_yaw(self): return self.righting_moment[2]

    def to_dict(self):
        return asdict(self)
