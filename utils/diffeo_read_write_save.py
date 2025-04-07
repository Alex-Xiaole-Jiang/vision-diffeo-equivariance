import torch as t
from dataclasses import dataclass, field

@dataclass
class DiffeoConfig:
    """Configuration for diffeomorphism parameters"""
    resolution: int = 192
    x_range: list[int] = field(default_factory=lambda: [0, 3])
    y_range: list[int] = field(default_factory=lambda: [0, 3])
    num_nonzero_params: int = 3
    strength: list[float] = field(default_factory=lambda: [0.1])
    num_diffeo_per_strength: int = 10


def read_diffeo_from_path(path: str, device = t.device("cpu")):
   diffeo_param_dict = t.load(path, weights_only=False, map_location = device)
   diffeo_param = diffeo_param_dict['AB']
   inv_diffeo_param = diffeo_param_dict['inv_AB']
   diffeo_strenghts = diffeo_param_dict['diffeo_config']['strength']
   num_of_diffeo_per_strength = diffeo_param_dict['diffeo_config']['num_diffeo_per_strength']
   return diffeo_param, inv_diffeo_param, diffeo_strenghts, num_of_diffeo_per_strength