import numpy as np

def serialize(msg_dict):
    ret = {}
    for key, value in msg_dict.items():
        if isinstance(value, (np.ndarray)):
            ret[key] = value.tolist()
        elif isinstance(value, (int, float, str, bool, list)):
            ret[key] = value
        else:
            raise TypeError(f"Unsupported type for key '{key}': {type(value)}")
    return ret