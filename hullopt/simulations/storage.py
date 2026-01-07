import pickle
import os
from .result import Result
from dataclasses import asdict
from typing import Dict, Tuple, Any
from hullopt.hull.hull import Params as hullParams
from dataclasses import dataclass, is_dataclass


class InputParameters:
    def __init__(self, *args):
        self._sources = args

    def __getattr__(self, name):
        for obj in self._sources:
            if hasattr(obj, name):
                return getattr(obj, name)

        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
    def to_dict(self):

        combined_data = {}

        for source in reversed(self._sources):

            if isinstance(source, dict):
                combined_data.update(source)
            
            elif is_dataclass(source):
                combined_data.update(asdict(source))
            elif hasattr(source, "__dict__"):
                combined_data.update(source.__dict__)
                
        return combined_data

class ResultStorage:
    def __init__(self, filepath: str = "gp_data.pkl"):
        self.filepath = filepath
        self.data: Dict[Tuple, Tuple[float, float, float]] = self._load_all()

    def _load_all(self) -> Dict:
        """
        Reads the pickle file sequentially. 
        The file is now a stream of separate pickle objects, not one big dict.
        """
        data = {}
        if os.path.exists(self.filepath):
            with open(self.filepath, "rb") as f:
                while True:
                    try:
                        entry = pickle.load(f)
                        key, val = entry
                        data[key] = val
                    except EOFError:

                        break
                    except pickle.UnpicklingError:
                        print(f"Warning: Corrupt data found in {self.filepath}")
                        break
                    
        return data

    def _append_to_file(self, key: Tuple, value: Any):
        """
        Appends a SINGLE entry to the end of the file.
        This is instant (O(1)), regardless of file size.
        """
        assert type(key) is tuple, "Key must be a tuple"

        with open(self.filepath, "ab") as f:  

            pickle.dump((key, value), f)




    def store(self, result_obj: 'Result', sim_params: Any, hull: Any) -> None:

        res_dict = result_obj.to_dict()
        params = InputParameters(sim_params, hull.params)


        param_dict = params.to_dict()



        target_val = res_dict.pop('righting_moment')


        if 'scene' in res_dict:
            del res_dict['scene']
            
        merged_data = {**res_dict, **param_dict}
        key_tuple = tuple(sorted(merged_data.items()))
        

        self.data[key_tuple] = target_val
        self._append_to_file(key_tuple, target_val)
        # print(f"Stored result for params: {param_dict}")
