#%%

import torch

from utils.diffeo_container import band_limited_sparse_transform_amplitude, DiffeoConfig
from utils.inverse_diffeo import find_param_inverse

import os
import logging

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

job_id = os.getenv('SLURM_JOB_ID')
logging.basicConfig(
      filename=f'slurm_{job_id}.err',
      level=logging.INFO,
      format='%(asctime)s - %(levelname)s - %(message)s'
  )
logging.info(f'device is {device}')

torch_seed = 37
torch.manual_seed(torch_seed)

DiffeoSavePath = '/vast/xj2173/diffeo/scratch_data/inv_diffeo_param/'
os.makedirs(DiffeoSavePath, exist_ok=True)

diffeo_config = DiffeoConfig(
    resolution = 128,
    x_range = [0, 5],
    y_range = [0, 5],
    num_nonzero_params = 3,
    strength = [0, 0.00001, 0.00002, 0.00005, 0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01],
    num_diffeo_per_strength = 20
)

A_list = []
B_list = []

for strength in diffeo_config.strength:
    A,B = band_limited_sparse_transform_amplitude(
        x_range = diffeo_config.x_range,
        y_range = diffeo_config.y_range,
        num_of_terms = diffeo_config.num_nonzero_params,
        diffeo_amp = strength,
        num_of_diffeo = diffeo_config.num_diffeo_per_strength
        )
    A_list.append(A)
    B_list.append(B)

AB = torch.stack([torch.cat(A_list), torch.cat(B_list)])

inv_param, inv_loss_hist, _ = find_param_inverse(AB, num_epochs = 1000, resolution = diffeo_config.resolution, device = device, disable_tqdm_log = False)
logging.info(inv_loss_hist)

name_of_file = (f'{diffeo_config.x_range[0]}-{diffeo_config.x_range[1]}-'
                f'{diffeo_config.y_range[0]}-{diffeo_config.y_range[1]}-'
                f'{diffeo_config.num_nonzero_params}-'
                f'{len(diffeo_config.strength)}-'
                f'{diffeo_config.num_diffeo_per_strength}-'
                f'{max(diffeo_config.strength)}')

torch.save({
    'diffeo_config': diffeo_config.__dict__,
    'AB': AB,
    'inv_AB': inv_param
}, os.path.join(DiffeoSavePath, name_of_file))