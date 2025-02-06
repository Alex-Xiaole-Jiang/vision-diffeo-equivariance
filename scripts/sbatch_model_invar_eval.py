#%%
import torch as t
import torch.nn.functional as F
import torchvision as tv
from torchvision.transforms import v2

from utils.diffeo_container import diffeo_container
from utils.distortion import create_grid_sample

import logging
import os
import sys

from tqdm import tqdm

device = t.device("cuda") if t.cuda.is_available() else t.device("cpu")

torch_seed = 37
t.manual_seed(torch_seed)

logging_to_print = False
# setting up path and config
ImageNet_path = '/imagenet'
num_of_images = 100

diffeo_Readpath = '/scratch/xj2173/diffeo_data/inv_diffeo_param/0-5-0-5-3-11-20-0.01'
# denoiser_Readpath = '/vast/xj2173/diffeo/scratch_data/Unet_weights/1667k/model_weights_39.pth'

save_path = f'/vast/xj2173/diffeo/scratch_data/steering/ENV2_s_no_steering/'
ref_path = f'/vast/xj2173/diffeo/scratch_data/steering/ENV2_s_no_steering/reference/'



# setting up helper function
def setup_logging(logging_debug = False):
  if logging_debug:
    logging.basicConfig(
      stream = sys.stdout,
      level=logging.DEBUG,
      format='%(asctime)s - %(levelname)s - %(message)s'
      )
  if not logging_debug:  
    job_id = os.getenv('SLURM_JOB_ID')
    logging.basicConfig(
        filename=f'slurm_{job_id}.err',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
  logging.info('Logger initialized')

def get_model(device = t.device('cpu')):
  model = tv.models.efficientnet_v2_s(weights = tv.models.EfficientNet_V2_S_Weights.DEFAULT).to(device)
  model_input_res = 384
  # ENV2 = tv.models.efficientnet_v2_s().to(device) # random initialization
  model.eval();
  for param in model.parameters():
      param.requires_grad = False
  return model.to(device), model_input_res

def get_inference_transform():
  inference_transform = tv.models.EfficientNet_V2_S_Weights.IMAGENET1K_V1.transforms()
  inference_transforms = v2.Compose([
      lambda x: x.convert('RGB'),
      inference_transform,
  ])
  return inference_transforms

def get_ImageNet(transforms = None, batch_size = 1, shuffle = False):
  dataset = tv.datasets.ImageNet(ImageNet_path, split = 'val', transform = transforms)
  dataloader = t.utils.data.DataLoader(dataset, batch_size = batch_size, shuffle = shuffle)
  return dataset, dataloader

def read_diffeo_from_path(path: str, device = t.device("cpu")):
   diffeo_param_dict = t.load(path, weights_only=False, map_location = device)
   diffeo_param = diffeo_param_dict['AB']
   inv_diffeo_param = diffeo_param_dict['inv_AB']
   diffeo_strenghts = diffeo_param_dict['diffeo_config']['strength']
   num_of_diffeo_per_strength = diffeo_param_dict['diffeo_config']['num_diffeo_per_strength']
   return diffeo_param, inv_diffeo_param, diffeo_strenghts, num_of_diffeo_per_strength

# where the code starts

setup_logging(logging_to_print)
logging.info(f'using device {device}')

model, input_res = get_model(device)

inf_transf = get_inference_transform()
dataset, dataloader = get_ImageNet(transforms = inf_transf, batch_size = 1, shuffle = True)
data_iter = iter(dataloader)

logging.info('model & dataset setup finished')
#%%
AB, inv_AB, diffeo_strengths, num_of_diffeo_per_strength = read_diffeo_from_path(diffeo_Readpath, device = t.device('cpu'))
diffeo_grids = create_grid_sample(input_res, input_res, AB[0], AB[1])
diffeos = diffeo_container(input_res, input_res, [diffeo_grids], device=device)
assert len(diffeo_strengths) * num_of_diffeo_per_strength == diffeo_grids.shape[0]
logging.info(f'diffeo strengths are {diffeo_strengths}')
logging.info('parameterized diffeo & inverse loaded')
#%%
val_images = t.cat([next(data_iter)[0] for i in range(num_of_images)], dim = 0).to(device)
ref_output = model(val_images)
t.save(ref_output, ref_path + f'val_image-first-{num_of_images}-images-output.pt')
logging.info('reference output computed & saved')
# layers = list(range(3,44)) + [46] + [49]
#%%
for i, image in enumerate(tqdm(val_images)):
  file_prefix = f'{len(diffeo_strengths)}-{num_of_diffeo_per_strength}-5-5-3-{max(diffeo_strengths)}-{input_res}-{input_res}_image-{i:04d}'
  
  # get a list of shape [strength * diffeo (batch), channel, x, y]
  distorted_list = diffeos(image.repeat(num_of_diffeo_per_strength * len(diffeo_strengths), 1,1,1), in_inference = True)
    
  with t.no_grad(): output = model(distorted_list)
  # steered has shape [layer, strength, diffeo, -1]
  output = t.reshape(output, (len(diffeo_strengths), num_of_diffeo_per_strength, -1))
  t.save(output, save_path + file_prefix +'.pt')
  
  # print(f'{i+1}th image completed', flush = True)