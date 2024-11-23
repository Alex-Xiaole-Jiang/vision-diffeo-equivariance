#%%
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision as tv
from torchvision.transforms import v2
from utils.diffeo_container import sparse_diffeo_container
from utils.get_model_activation import retrieve_layer_activation
from utils.inverse_diffeo import find_param_inverse

from model.Unet import UNet

import os
import logging

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

job_id = os.getenv('SLURM_JOB_ID')
logging.basicConfig(
    filename=f'slurm_{job_id}.err',  # %j will be replaced with SLURM job ID
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logging.info("JOB STARTED")
logging.info(f'uses {device}')

def get_inference_transform():
  inference_transform = tv.models.EfficientNet_V2_S_Weights.IMAGENET1K_V1.transforms()
  inference_transforms = v2.Compose([
      lambda x: x.convert('RGB'),
      inference_transform,
  ])
  return inference_transforms

def get_diffeo_container(x_range, y_range, 
                         num_of_terms, 
                         diffeo_strength_list = None, 
                         num_of_didffeo = None, 
                         res = None,
                         device = torch.device('cpu')) -> sparse_diffeo_container:
  
  diffeo_container = sparse_diffeo_container(res, res)
  for strength in diffeo_strength_list:
      diffeo_container.sparse_AB_append(x_range, y_range, num_of_terms, strength, num_of_didffeo)
  diffeo_container.get_all_grid()
  diffeo_container.to(device)

  return diffeo_container

def get_inv_diffeo_container(diffeo_container: sparse_diffeo_container, 
                             res = 224, 
                             device = torch.device("cpu"), 
                             **kwargs) -> tuple[sparse_diffeo_container, list, list]:
  
  AB = torch.stack([(diffeo_container.A[0]), (diffeo_container.B[0])])
  [A_inv, B_inv], loss_hist, mag_hist = find_param_inverse(AB, device = device, resolution=res, **kwargs)
  inv_diffeos = sparse_diffeo_container(res, res, 
                                      A = [A_inv.cpu()], B = [B_inv.cpu()])
  inv_diffeos.get_all_grid()
  inv_diffeos.to(device)

  return inv_diffeos, loss_hist, mag_hist

def get_dataset_subset(dataset, num_image_per_class):  
  prev = 0
  start = 0
  end = None
  class_range_index = []
  for i, target in enumerate(dataset.targets):
      if target != prev:
          prev = target
          end = start
          start = i
          class_range_index.append([end, start])
  class_range_index.append([start, i])

  subset_index = []
  test_index = []
  for index_range in class_range_index:
      subset_index += list(range(index_range[0], index_range[0]+num_image_per_class))
      test_index += [index_range[0]+num_image_per_class+1]

  dataset_subset = torch.utils.data.Subset(dataset, subset_index)
  testset_subset = torch.utils.data.Subset(dataset, test_index)
  return dataset_subset, testset_subset

def get_activation_of_training_image(model, images, layer_number):
  with torch.no_grad():
    activation, _  = retrieve_layer_activation(model, images, [layer_number])
  return activation[layer_number]

def get_diffeo_blurred_image(images, diffeos, inv_diffeos, number_of_samples):
  indices = torch.randperm(len(diffeos[0]))[:number_of_samples]
  blurr_list = []
  for image in images:
     distorted = nn.functional.grid_sample(image.expand(number_of_samples,-1,-1,-1), diffeos[0][indices], align_corners = True)
     # distorted = diffeos(image.expand(number_of_samples, -1,-1,-1))
     # blurred = inv_diffeos(distorted)
     blurred = nn.functional.grid_sample(distorted, inv_diffeos[0][indices], align_corners = True)
     blurr_list.append((blurred).squeeze())
  return torch.stack(blurr_list)


# Main scirpt start
logging.info('get model and subset dataset')
ImageNet_path = '/imagenet'
number_of_training_image_per_class = 5
batch_size = 200
number_of_channels_to_use = 8
activation_layer = 5
logging.info(f'total a total of {number_of_training_image_per_class * 1000} training images')

# model and dataset
vision_model = tv.models.efficientnet_v2_s(weights = tv.models.EfficientNet_V2_S_Weights.DEFAULT).to(device)
vision_model.eval();
transform = get_inference_transform()
dataset = tv.datasets.ImageNet(ImageNet_path, split = 'train', transform = transform)
sub_dataset, test_dataset = get_dataset_subset(dataset, number_of_training_image_per_class)
dataloader = torch.utils.data.DataLoader(sub_dataset, batch_size = batch_size, shuffle = True)
testLoader = torch.utils.data.DataLoader(test_dataset, batch_size = batch_size, shuffle = True)

logging.info('got model and dataset/loader')
logging.info('get inv/diffeo')

x_range = y_range = [0,3]
number_of_non_zero_diffeo_parameter = 3
diffeo_strength = [0.1]
total_number_of_random_diffeos = 200
diffeo_sample_num = 8
blurry_resolution = 192

diffeos = get_diffeo_container(x_range, 
                                y_range, 
                                number_of_non_zero_diffeo_parameter, 
                                diffeo_strength,
                                total_number_of_random_diffeos,
                                blurry_resolution,
                                device = device)

inv_diffeos,_,_ = get_inv_diffeo_container(diffeos, res = blurry_resolution, device = device)

logging.info('got inv/diffeos')
logging.info('initialize denoising autoencoder and training routine')

model_config = {
        'in_channels': 1,
        'out_channels': 1,
        'initial_filters': 32,
        'depth': 1,
        'conv_kwargs': {'kernel_size': 5, 'padding': 2},
        'upsample_kwargs': {'mode': 'bilinear', 'align_corners': True},
        'num_conv_in_resid':[2,2],
        'num_resid_blocks':[5,5],
        'num_conv_in_skip':[2]
    }
denoiser = UNet(model_config)
denoiser.to(device)


logging.info(f'model config: {model_config}')
model_parameter_num = sum([p.numel() for p in denoiser.parameters()])
logging.info(f'model with {model_parameter_num} number of parameters')

model_weights_dir = f'/vast/xj2173/diffeo/scratch_data/Unet_weights/{int(round(model_parameter_num/1000))}k/'
os.makedirs(model_weights_dir, exist_ok=True)

loading_model = False
if loading_model:
    model_save = torch.load(model_weights_dir+'model_weights_{i}.pth', map_location=torch.device('cpu'))
    denoiser = UNet(model_save['model_config'])
    denoiser.load_state_dict(model_save['model_state_dict'])
    denoiser = denoiser.to(device)


learning_rate = 0.001 # default, previously 0.05 too large
weight_decay = 0.01
epochs = 40
loss_fn = nn.MSELoss()

optimizer = optim.AdamW(denoiser.parameters(),
                              lr=learning_rate,
                              weight_decay=weight_decay)

logging.info(f'training {epochs} epochs with lr:{learning_rate}, weight_decay:{weight_decay}')
logging.info('training start')

total_image_num = batch_size * len(dataloader)
trai_loss_hist = []
test_loss_hist = []

for i in (range(epochs)):
  trai_loss_accumulant = 0
  test_loss_accumulant = 0

  for images, _ in dataloader:
    logging.debug('start sub-epoch loop')
    train_img = images.to(device)
    train_img = get_activation_of_training_image(vision_model, train_img, activation_layer)
    indices = torch.randperm(train_img.shape[1])[:number_of_channels_to_use]
    train_img = train_img[:,indices]

    blurred = get_diffeo_blurred_image(train_img, diffeos, inv_diffeos, diffeo_sample_num)
    logging.debug('data prep finished')

    for target, blurry in zip(train_img, blurred):
      optimizer.zero_grad()
      target = target.expand(len(blurry), -1, -1, -1)
      diff, channel, x, y = target.shape
      target = torch.reshape(target ,(diff*channel, 1, x, y))
      blurry = torch.reshape(blurry ,(diff*channel, 1, x, y))
      output = denoiser(blurry)
      loss = loss_fn(output, target)
      loss.backward()
      optimizer.step()
      trai_loss_accumulant += loss.item()
    logging.debug('sub-epoch loop finished')

  logging.info('calculating training loss')
  with torch.no_grad():
    for test_images, _ in testLoader:
      test_img = test_images.to(device)
      test_img = get_activation_of_training_image(vision_model, test_img, activation_layer)
      blurred = get_diffeo_blurred_image(train_img, diffeos, inv_diffeos, diffeo_sample_num)
      for target, blurry in zip(train_img, blurred):
        target = target.expand(len(blurry), -1, -1, -1)
        diff, channel, x, y = target.shape
        target = torch.reshape(target ,(diff*channel, 1, x, y))
        blurry = torch.reshape(blurry ,(diff*channel, 1, x, y))
        output = denoiser(blurry)
        loss = loss_fn(output, target)
        test_loss_accumulant += loss.item()

  logging.info(f'test loss is {test_loss_accumulant/(batch_size * len(testLoader))} at epoch {i}')
  logging.info(f'training loss is {trai_loss_accumulant/total_image_num} at epoch {i}')
  test_loss_hist.append(test_loss_accumulant/(batch_size * len(testLoader)))
  trai_loss_hist.append(trai_loss_accumulant/total_image_num)

  torch.save({'model_config': model_config, 
              'model_state_dict': denoiser.state_dict(),
              'train_loss': trai_loss_hist,
              'test_loss': test_loss_hist},
              model_weights_dir + f'model_weights_{i}_new.pth')