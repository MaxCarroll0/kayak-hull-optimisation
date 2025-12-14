import pickle
import os
from .result import Result
from dataclasses import asdict
from typing import Dict, Tuple, Any

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
        with open(self.filepath, "ab") as f:  
            pickle.dump((key, value), f)

    def store(self, result_obj: 'Result', params: Any):
        res_dict = result_obj.to_dict()
        

        if hasattr(params, '__dataclass_fields__'):
            param_dict = asdict(params)
        else:
            param_dict = vars(params)


        target_val = res_dict.pop('righting_moment')


        if 'scene' in res_dict:
            del res_dict['scene']
            
        merged_data = {**res_dict, **param_dict}
        key_tuple = tuple(sorted(merged_data.items()))
        

        self.data[key_tuple] = target_val
        self._append_to_file(key_tuple, target_val)