from dataclasses import dataclass, asdict
from trimesh import Scene
from typing import Tuple


@dataclass
class Result:
    """
    Simulation resuts: These only make sense if the hull floats at the given angle. Otherwise they might fail to converge or output garbage results.
    
    righting_moment - Nm (float, float, float): angular forces exerted on the hull by buoyancy & fluid flow (note 3 dimensions x,y,z)
    reserve_buoyancy - kg (float): maximum extra lift possible by water displaced by pushing the hull further underwater (generally, the point of downflooding)
    reserve_buoyancy_hull - kg (float): the greatest reserve buoyancy that the submerged portion of hull contributes (i.e. reserve buoyancy excluding air pockets within the hull)
    scene - Trimesh.Scene: scene containing the tilted hull & waterline for viewing with scene.show()
    cost - float: Simulation cost (accounting for # of iterations, and discretisation). Note: does not account for (hardware-dependent) time taken to complete
    """


    righting_moment: Tuple[float, float, float]
    reserve_buoyancy: float
    reserve_buoyancy_hull: float
    scene: Scene
    cost: float

    def righting_moment_heel(self): return self.righting_moment[0]
    def righting_moment_pitch(self): return self.righting_moment[1]
    def righting_moment_yaw(self): return self.righting_moment[2]

    def to_dict(self):
        return asdict(self)
