import pandas as pd # Optional, but good for visualizing
import numpy as np
import pickle
from typing import Tuple, Dict, Any, List


def get_category_heuristic(param_name: str) -> str:
    """Fallback logic to guess category if not in dictionary."""
    param_lower = param_name.lower()
    
    # Simple heuristics
    if 'speed' in param_lower or param_lower == 'v':
        return 'speed'
    if any(x in param_lower for x in ['angle', 'heel', 'yaw', 'leeway']):
        return 'angles'
    
    # Default fallback
    return 'shape'

def default_param_categories() -> Dict[str, str]:
    return {'heel': 'angles', 'length': 'shape', 'beam': 'shape', 'density': 'shape', 'draft': 'shape', 'section_shape_exponent': 'shape'}

def load_simulation_data(filepath: str, user_categories: Dict[str, str] = None) -> Tuple[np.ndarray, np.ndarray, Dict[str, List[int]]]:
    
    category_map = default_param_categories()
    if user_categories:
        category_map.update(user_categories)

    raw_inputs = []
    raw_outputs = []

    with open(filepath, 'rb') as f:
        while True:
            try:
                row_data = pickle.load(f)
                
                input_tuple = row_data[0]
                output_tuple = row_data[1]
                
                raw_inputs.append(dict(input_tuple))
                raw_outputs.append([float(x) for x in output_tuple])
                
            except EOFError:
                break
            except Exception as e:
                print(f"Skipping corrupted line: {e}")
                continue
    
    if not raw_inputs:
        print("No data loaded.")
        return np.array([]), np.array([]), {}

    feature_order = list(raw_inputs[0].keys())
    print(feature_order)
    
    col_map = {'speed': [], 'angles': [], 'shape': []}
    
    print("-" * 40)
    print(f"Found {len(feature_order)} parameters in file. Mapping categories:")
    
    for idx, key in enumerate(feature_order):
        
        if key in category_map:
            cat = category_map[key]
            source = "Dictionary"

        else:
            cat = get_category_heuristic(key)
            source = "Heuristic"
            
        if cat not in col_map:

            print(f"  [Warning] Unknown category '{cat}' for '{key}'. Defaulting to 'shape'.")
            cat = 'shape'


        col_map[cat].append(idx)
        print(f"  Col {idx}: '{key}' -> {cat} ({source})")

    print("-" * 40)


    X_list = []
    for d in raw_inputs:
        print(d)

        row = [float(d[k]) for k in feature_order]
        X_list.append(row)

    X = np.array(X_list, dtype=np.float64)
    y = np.array(raw_outputs, dtype=np.float64)
    print(col_map)
    return X, y, col_map