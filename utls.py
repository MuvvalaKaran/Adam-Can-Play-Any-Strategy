import json
import numpy as np

class NpEncoder(json.JSONEncoder):
    """
     Custom JSON encoder for NumPy data types.

     This encoder handles the serialization of NumPy data types to ensure they can be converted to JSON format.
     Specifically, it converts NumPy integers to Python integers, NumPy floating-point numbers to Python floats,
     and NumPy arrays to Python lists.

     I use this to serialize NumPy data types to JSON format, which is useful for platting graph using the D3.js package.
    """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)
