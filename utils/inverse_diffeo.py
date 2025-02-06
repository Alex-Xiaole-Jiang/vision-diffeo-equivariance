import torch as t
import torch.nn as nn 
import torch.nn.functional as F

from utils.distortion import compose_diffeo_from_left, create_grid_sample, get_id_grid
from tqdm import tqdm

torch_seed = 37
t.manual_seed(torch_seed)


class add_bias_to_AB(nn.Module):
  '''
  Neural network module that adds learnable bias to AB coefficients with frequency scaling.
  
  Args:
    AB: Input tensor of AB coefficients
    extra_freq_scaling: Factor to scale the padding size of higher frequencies
  
  Properties:
    result_magnitude: Sum of absolute values for the final result (A and B)
    bias_magnitude: Sum of absolute values for the learned bias (A and B)
    original_magnitude: Sum of absolute values for the original AB coefficients
  
  Notes:
    - Initializes bias randomly scaled by the number of elements
    - Pads the input AB tensor with zeros based on frequency scaling factor
'''
  def __init__(self, AB, extra_freq_scaling = 1):
    # grid should have the shape of a grid_sample grid, i.e. (Channel, X, Y, 2)
    super().__init__()
    _, _, x_cutoff, y_cutoff = AB.shape
    self.AB = F.pad(AB,(0,(extra_freq_scaling) * y_cutoff, 0, (extra_freq_scaling) * x_cutoff),mode = 'constant', value = 0)
    # self.bias = nn.Parameter(self.AB.abs().sum() * (t.randint_like(self.AB, 1) - 1/2)/self.AB.numel())
    # self.bias = nn.Parameter(t.rand_like(self.AB))
    self.bias = nn.Parameter(t.rand_like(self.AB)/self.AB.numel())
    self.result = AB
  def forward(self):
    self.result = self.bias + self.AB
    return self.result
  @property
  def result_magnitude(self):
    return self.result[0].detach().abs().sum(), self.result[1].detach().abs().sum()
  @property
  def bias_magnitude(self):
    return self.bias[0].detach().abs().sum(), self.bias[1].detach().abs().sum()
  @property
  def original_magnitude(self):
    return self.AB[0].detach().abs().sum(), self.AB[1].detach().abs().sum()
  
# AB_original = t.cat([t.stack(sparse_diffeos.A), t.stack(sparse_diffeos.B)])
def find_param_inverse(AB: t.Tensor, 
                       extra_freq_scaling = 5, 
                       num_epochs = 500,
                       resolution = 384, 
                       device = t.device('cpu'),
                       disable_tqdm_log = True) -> t.Tensor:
  '''
  Finds inverse parameters for a diffeomorphism using gradient descent.
  
  Args:
    AB: Input tensor [A,B] representing diffeomorphism parameters
    extra_freq_scaling: Factor for frequency padding in parameter space
    num_epochs: Number of optimization iterations
    resolution: the resolution to learn the diffeo
    device: PyTorch device for computation
  
  Returns:
  Tuple containing:
  - Inverse parameters [A_inverse, B_inverse] as tensor
  - History of inverse loss values
  - History of parameter magnitudes
  
  Notes:
  - Uses Adagrad optimizer with MSE loss
  - Grid size given by resolution
  - Includes regularization based on parameter magnitudes
  - Small bias values (<1e-7) are zeroed every 50 epochs
  '''
  inv_loss_hist = []
  AB_mag = []

  grid = create_grid_sample(resolution, resolution, AB[0], AB[1]).to(device)
  AB = add_bias_to_AB(-AB.to(device), extra_freq_scaling = extra_freq_scaling).to(device)
  loss_fn = nn.MSELoss()

  optimizer = t.optim.Adagrad(AB.parameters(), lr = 0.1)
  
  id_grid = get_id_grid(resolution, resolution, device)
  
  for epoch in tqdm(range(num_epochs), disable=disable_tqdm_log):
      optimizer.zero_grad()
      new_AB = AB()
      with t.device(device):
          inv_grid = create_grid_sample(resolution, resolution, new_AB[0], new_AB[1]).to(device)
      un_distorted = compose_diffeo_from_left(inv_grid, grid)
      unreg_loss = loss_fn(un_distorted, id_grid.expand(len(un_distorted),-1,-1,-1))
      loss = unreg_loss * (1 + 0.1 * AB.result.abs().sum()/AB.AB.abs().sum())
      loss.backward()
      optimizer.step()
      # scheduler.step(loss)

      if (epoch) % 50 == 0:
            with t.no_grad(): 
              inv_loss_hist.append(unreg_loss.item())
              AB_mag.append(AB.result.abs().sum().item()/2)
              AB.bias[t.abs(AB.bias) < 1e-7] = 0
            # print(loss.item())
  
  return AB.result.detach() , inv_loss_hist, AB_mag



# AB_original = t.cat([t.stack(sparse_diffeos.A), t.stack(sparse_diffeos.B)])
def find_img_inverse(distorted_img: t.Tensor, 
                     target_img:t.Tensor, 
                     AB: t.Tensor, 
                     extra_freq_scaling = 5, 
                     num_epochs = 500, 
                     device = t.device('cpu'), 
                     grid_sample_kwargs = {}) -> t.Tensor:
  '''
  input:
    distorted_img: distorted image
    target_img: image we want the distorted image to look like after applying diffeo
    parameters of the diffeo as a t.Tensor([A,B])
  output: 
    t.Tensor([A_inverse, B_inverse]) that gives the inverse of the diffeo
  the optimizer is Adagrad
  '''
  inv_loss_hist = []
  AB_mag = []

  batch_size, channel_num, x_res, y_res = distorted_img.shape

  AB = add_bias_to_AB(-AB.to(device), extra_freq_scaling = extra_freq_scaling).to(device)
  loss_fn = nn.MSELoss()

  optimizer = t.optim.Adagrad(AB.parameters(), lr = 0.1)
  
  
  for epoch in tqdm(range(num_epochs)):
      optimizer.zero_grad()
      new_AB = AB()
      with t.device(device):
          inv_grid = create_grid_sample(x_res, y_res, new_AB[0], new_AB[1]).to(device)
      un_distorted = F.grid_sample(distorted_img, inv_grid.expand(batch_size, -1, -1, -1), 
                                   align_corners=True, **grid_sample_kwargs)
      unreg_loss = loss_fn(un_distorted, target_img)
      loss = unreg_loss * (1 + 0.1 * AB.result.abs().sum()/AB.AB.abs().sum())
      loss.backward()
      optimizer.step()
      # scheduler.step(loss)

      if (epoch) % 50 == 0:
            with t.no_grad(): 
              inv_loss_hist.append(unreg_loss.item())
              AB_mag.append(AB.result.abs().sum().item()/2)
              AB.bias[t.abs(AB.bias) < 1e-7] = 0
            # print(loss.item())
  
  return AB.result.detach() , inv_loss_hist, AB_mag


def find_inv_grid(flow_grid, mode ='bilinear', learning_rate = 0.001, epochs = 10000, early_stopping = True, align_corners = True):
  device = flow_grid.device
  batch, x_length, y_length, _ = flow_grid.shape
  x = t.linspace(-1, 1, steps = x_length)
  y = t.linspace(-1, 1, steps = y_length)
  X, Y = t.meshgrid(x, y, indexing='ij')
  reference = t.stack((X, Y, X * Y), dim=0).unsqueeze(0).repeat(batch, 1, 1, 1).to(device)
  #, t.cos(2*math.pi*X) * t.cos(2*math.pi*Y)
  id_grid = t.stack((Y, X), dim=-1).unsqueeze(0).repeat(batch, 1, 1, 1).to(device)
  #2 * id_grid - flow_grid
  find_inv_model = add_bias_to_grid(id_grid).to(device)
  loss_fn = nn.MSELoss()
  optimizer = t.optim.SGD(find_inv_model.parameters(), lr = learning_rate)

  num_epochs = epochs
  loss_hist = []
  min_loss = 1e30
  early_stopping_count = 0
  for epoch in tqdm(range(num_epochs)):
    optimizer.zero_grad()
    output = find_inv_model()
    distort = t.nn.functional.grid_sample(reference, flow_grid, mode = mode, align_corners = align_corners)
    #inv_distort = t.nn.functional.grid_sample(reference, output, mode = mode)
    restored_left  = t.nn.functional.grid_sample(distort, output, mode = mode, align_corners = align_corners)
    #restored_right = t.nn.functional.grid_sample(inv_distort, flow_grid, mode = mode)
    left_loss = loss_fn(reference, restored_left)
    #right_loss = loss_fn(reference, restored_right)
    loss = left_loss #+ right_loss #+ (t.exp(t.abs(left_loss-right_loss)**2) - 1)
    #loss =  left_loss + right_loss
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 20 == 0:
          loss_hist.append(loss.item())
          if loss_hist[-1]/min_loss >= 1: 
            early_stopping_count += 1
            # print(f'Early stopping count: {early_stopping_count}')
          if loss_hist[-1] < min_loss: 
            min_loss = loss_hist[-1]
            early_stopping_count = 0
    if early_stopping and early_stopping_count >=10: break

  with t.no_grad():
    flow_grid_inverse_neural = find_inv_model().detach().clone()
  
  del find_inv_model

  return flow_grid_inverse_neural, loss_hist, epoch
