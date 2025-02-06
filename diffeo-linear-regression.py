import torch 
import torch.nn.functional as F
from utils.distortion import create_grid_sample

diffeo_path = '/home/xj2173/vast/diffeo/scratch_data/inv_diffeo_param/0-5-0-5-3-15-20'

diffeo_data = torch.load(diffeo_path, map_location = torch.device('cpu'), weights_only=False)
diffeo_param = diffeo_data['AB'][:,-1:,:,:]
grid = create_grid_sample(128, 128, *diffeo_param)
print(grid.shape)