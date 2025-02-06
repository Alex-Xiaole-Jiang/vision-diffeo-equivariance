#%%
import torch as t
import torch.nn.functional as F
import torchvision as tv
from torchvision.transforms import v2

from utils.diffeo_container import sparse_diffeo_container, diffeo_container
from utils.distortion import create_grid_sample
from utils.get_model_activation import get_flatten_children

from model.EDSR.EDSR import SingleScaleEDSR, SingleScaleEDSRConfig

import logging
import os
import sys

from tqdm import tqdm
import dacite

import torch._dynamo
torch._dynamo.config.suppress_errors = True
torch.set_grad_enabled(False)

device = t.device("cuda") if t.cuda.is_available() else t.device("cpu")

torch_seed = 37
t.manual_seed(torch_seed)

logging_to_print = False
# setting up path and config
ImageNet_path = '/imagenet'
num_of_images = 100

diffeo_Readpath = '/scratch/xj2173/diffeo_data/inv_diffeo_param/0-5-0-5-3-15-20-0.65'
denoiser_Readpath = '/vast/xj2173/diffeo/scratch_data/Unet_weights/1667k/model_weights_39.pth'

# steering_layers = list(range(3,44))
steering_layers = range(3,14)
# steering_layers = [3]
use_denoiser = True
if use_denoiser: 
  denoised = '_denoised'
else:
  denoised = ''

save_path = f'/vast/xj2173/diffeo/scratch_data/steering/ENV2_s_NN{denoised}/'
ref_path = f'/vast/xj2173/diffeo/scratch_data/steering/ENV2_s_NN{denoised}/reference/'



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

def load_denoiser(path: str, device = t.device("cpu")):
   model_spec = t.load(path, weights_only=False, map_location = device)
   model_config = model_spec['model_config']
   model_state_dict = model_spec['model_state_dict']
   for key in list(model_state_dict.keys()):
     if '_orig_mod.' in key:
       model_state_dict[key.replace('_orig_mod.', '')] = model_state_dict.pop(key)
   model_config = dacite.from_dict(data_class = SingleScaleEDSRConfig, 
                                   data = model_config)
   model = SingleScaleEDSR(model_config)
   model.load_state_dict(model_state_dict)
   model.eval()
   for param in model.parameters():
     param.requires_grad = False
   return t.compile(model).to(device)

def denoiser_inference(denoiser: t.nn.Module, input, target_size=192):
  # inference by channel to decrease memory footprint
  B, C, H, W = input.shape
  if H != target_size or W != target_size:
    input = F.interpolate(input, (target_size, target_size), mode='bilinear', align_corners=True)
   
  outputs = []
  for c in range(C):
    outputs.append(denoiser(input[:, c:c+1]))
   
    out = torch.cat(outputs, dim=1)
   
  if H != target_size or W != target_size:
    out = F.interpolate(out, (H, W), mode='bilinear', align_corners=True)
  return out

def get_steering_layer_shapes(model, layers, input_resolution):
  # get layer size
  input_size = (1,3,input_resolution,input_resolution) # standard ImageNet size
  dummy_input = t.rand(*input_size).to(device)
  hooks = []
  layer_shapes = []

  def get_output_size_hook(module, input, output):
    x, y = output.shape[-2:]
    layer_shapes.append((x,y))
    pass

  for layer in layers:
    hooks.append(layer.register_forward_hook(get_output_size_hook))
  
  with t.no_grad(): model(dummy_input)
  for hook in hooks: hook.remove()

  return layer_shapes

def get_steering_inv(diffeo_param, layer_shapes):
  down_sampled_inv_diffeo = []
  for layer_shape in layer_shapes:
    x, y = layer_shape
    inv_grid = create_grid_sample(x, y, diffeo_param[0], diffeo_param[1])
    inv_diffeo = diffeo_container(x, y, [inv_grid])
    down_sampled_inv_diffeo.append(inv_diffeo)
  return down_sampled_inv_diffeo

def inv_diff_hook(inverse_diffeo, denoiser = None, batch_size = None):
    if not isinstance(inverse_diffeo, diffeo_container):        
        raise Exception('diffeo is not a diffeo_container')
    def hook(module, input, output):
        # print(output.shape)
        # print(f"Allocated: {t.cuda.memory_allocated() / 1e9:.2f}GB")
        # print(f"Max allocated: {torch.cuda.max_memory_allocated() / 1e9:.2f}GB")
        if batch_size is None:
        # normal situation
          output = inverse_diffeo(output, in_inference=True)
          if denoiser is None:
            return output
          if denoiser is not None:
            output = denoiser(output)
            return output
        if batch_size is not None:
        # stack in the batch dimension of the steered result
            output = t.unflatten(output, 0, (-1, batch_size))
            # print(output.shape)
            ref = output[0].clone()
            output = t.flatten(output, start_dim = 0, end_dim = 1)
            # print(output.shape)
            if denoiser is None:
              output = t.cat([output, inverse_diffeo(ref, in_inference = True)], dim = 0)
              return output
            if denoiser is not None:
              output = t.cat([output, denoiser(inverse_diffeo(ref, in_inference = True))], dim = 0)
              return output
    return hook

def steering_hooks(layers, inv_diffs, batch_size = None, denoiser = None):
# batch size = len(diffeo_strengths) * num_of_diffeo
  hooks = []
  for layer, inv_diff in zip(layers, inv_diffs):
    hooks.append(layer.register_forward_hook(inv_diff_hook(inv_diff, batch_size = batch_size, denoiser = denoiser)))
  return hooks
#%%
#
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
denoiser = load_denoiser(denoiser_Readpath, device=device)
if use_denoiser:
  denoiser_fn = lambda input: denoiser_inference(denoiser, input, target_size = 192)
else:
  denoiser_fn = None
logging.info(f'Denoiser status: {use_denoiser}')
#%%
val_images = t.cat([next(data_iter)[0] for i in range(num_of_images)], dim = 0).to(device)
ref_output = model(val_images)
t.save(ref_output, ref_path + f'val_image-first-{num_of_images}-images-output.pt')
logging.info('reference output computed & saved')
#%%
model_layer = get_flatten_children(model)
steering_model_layers = [model_layer[index] for index in steering_layers]
steering_layer_shapes = get_steering_layer_shapes(model, steering_model_layers, input_res)
steering_inv_diffeo = get_steering_inv(inv_AB, steering_layer_shapes)
for diffeo in steering_inv_diffeo: diffeo.to(device)
hooks = steering_hooks(steering_model_layers, 
                       steering_inv_diffeo, 
                       batch_size= diffeo_grids.shape[0], 
                       denoiser = denoiser_fn)

logging.info('hooks prepared')
# layers = list(range(3,44)) + [46] + [49]
#%%
for i, image in enumerate(tqdm(val_images)):
  file_prefix = f'{len(diffeo_strengths)}-{num_of_diffeo_per_strength}-4-4-3-{max(diffeo_strengths)}-{input_res}-{input_res}_image-{i:04d}_steered{denoised}'
  layers_in_string = '-'.join(str(num) for num in steering_layers)

  
  # get a list of shape [strength * diffeo (batch), channel, x, y]
  distorted_list = diffeos(image.repeat(num_of_diffeo_per_strength * len(diffeo_strengths), 1,1,1), in_inference = True)
    
  with t.no_grad(): steered = model(distorted_list)
  # steered has shape [layer, strength, diffeo, -1]
  steered = t.reshape(steered, (len(steering_layers)+1, len(diffeo_strengths), num_of_diffeo_per_strength, -1))
  t.save(steered, save_path + file_prefix + '_layer-' + layers_in_string +'.pt')
  
  # print(f'{i+1}th image completed', flush = True)

for hook in hooks: hook.remove()