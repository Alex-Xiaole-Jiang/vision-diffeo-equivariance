# %%
import numpy as np
import torch
import torch as t
import torch.nn as nn
import math
from tqdm import tqdm
#from torch_perlin_noise import rand_perlin_2d_octaves
# import matplotlib.pyplot as plt
#%%
def get_version(): print('version channel mixing')

def dense_transform_amplitude(x_length, 
                              y_length, 
                              truncate = False, 
                              amplitude = 1, 
                              rng = None, 
                              seed = 37, 
                              alpha = None):
  '''
    Returns x_length by y_length matrix where each element is random and has flattened norm of 1
    (except the 0,0 element).

    Args:
        x_length: Specifies the frequency bound in x dimension
        y_length: Specifies the frequency bound in y dimension
        truncate: Zeros already generated higher frequency components if True
        amplitude: Changes the norm
        rng: Random number generator instance
        seed: Random seed when rng is None
        alpha: Dirichlet distribution parameter

    Returns:
        Tuple of two matrices (A_nm, B_nm) with random elements
    '''
  if rng == None: rng = np.random.default_rng(seed = seed)

  A_sign = (rng.integers(2, size = x_length * y_length) - 0.5) * 2
  B_sign = (rng.integers(2, size = x_length * y_length) - 0.5) * 2

  if alpha == None:
    A_nm = amplitude * rng.dirichlet(np.ones(x_length * y_length - 1)) * A_sign
    B_nm = amplitude * rng.dirichlet(np.ones(x_length * y_length - 1)) * B_sign
  elif alpha != None:
    A_nm = amplitude * rng.dirichlet(alpha) * A_sign
    B_nm = amplitude * rng.dirichlet(alpha) * B_sign

  # A_nm = np.pad(A_nm, (1,0), 'constant').reshape(x_length, y_length)
  # B_nm = np.pad(B_nm, (1,0), 'constant').reshape(x_length, y_length)

  A_nm = A_nm.reshape(x_length, y_length)
  B_nm = B_nm.reshape(x_length, y_length)

  if truncate == False:
    return A_nm, B_nm

  A_nm = A_nm[0:truncate, 0:truncate]
  B_nm = B_nm[0:truncate, 0:truncate]

  return A_nm, B_nm
#%%
def band_limited_sparse_transform_amplitude(x_range: list, 
                                            y_range: list, 
                                            num_of_terms: int, 
                                            diffeo_amp = 1, 
                                            rng = None, 
                                            seed = 37, 
                                            num_of_diffeo = 1, 
                                            alpha = None) -> tuple[t.Tensor, t.Tensor]:
  '''
    Arguments:
      - x_range: the range of x frequency
      - y_range: the range of y frequency
      - num_of_terms: number of non-zero entries in the A,B matrix
      - diffeo_amp: the sum of the absolute value of the diffeo coefficient
      - rng: a rng used to shuffle
      - seed: if no rng is provided we use the seed to seed generation
      - num_of_diffeo: number of diffeos to generate
      - alpha: the Dirichlet distribution 'tilt'
    Returns A,B: 
      - A (torch.tensor): Has shape (num_of_diffeo, x_range[-1], y_range[-1]), 
        parameterizes diffeo coefficient for x-distortion
      - B (torch.tensor): Has shape (num_of_diffeo, x_range[-1], y_range[-1]), 
        parameterizes diffeo coefficient for y-distortion
  
  '''
  if rng == None: rng = np.random.default_rng(seed = seed)

  A = []
  B = []

  index_bank = [[i,j] for i in range(x_range[-1]) for j in range(y_range[-1]) 
                if i >= x_range[0] or j >= y_range[0]]
  index_bank = np.array(index_bank)

  for _ in range(num_of_diffeo):
    A_empty = np.zeros((x_range[-1], y_range[-1]))
    B_empty = np.zeros((x_range[-1], y_range[-1]))

    A_sign = (rng.integers(2, size = num_of_terms) - 0.5) * 2
    B_sign = (rng.integers(2, size = num_of_terms) - 0.5) * 2

    if alpha == None:
      A_nm = diffeo_amp * rng.dirichlet(np.ones(num_of_terms)) * A_sign
      B_nm = diffeo_amp * rng.dirichlet(np.ones(num_of_terms)) * B_sign
    elif alpha != None:
      A_nm = diffeo_amp * rng.dirichlet(alpha * np.ones(num_of_terms)) * A_sign
      B_nm = diffeo_amp * rng.dirichlet(alpha * np.ones(num_of_terms)) * B_sign
    
    
    index_list = [index_bank[i] for i in rng.choice(range(len(index_bank)),
                                                    num_of_terms, 
                                                    replace=False)]

    for i, index in enumerate(index_list):
      [x_index, y_index] = index
      A_empty[x_index, y_index] = A_nm[i]

    index_list = [index_bank[i] for i in rng.choice(range(len(index_bank)),
                                                    num_of_terms, 
                                                    replace=False)]

    for i, index in enumerate(index_list):
      [x_index, y_index] = index
      B_empty[x_index, y_index] = B_nm[i]

    A.append(t.Tensor(A_empty))
    B.append(t.Tensor(B_empty))

  return t.stack(A), t.stack(B)

#%%
def sparse_transform_amplitude(x_cutoff, 
                               y_cutoff, 
                               num_of_terms, 
                               diffeo_amp = 1, 
                               rng = None, 
                               seed = 37, 
                               num_of_diffeo = 1, 
                               alpha = None) -> tuple[t.Tensor, t.Tensor]:
  '''
    Returns A,B: 
      - A (torch.tensor): Has shape (num_of_diffeo, x_cutoff, y_cutoff), 
        parameterizes diffeo coefficient for x-distortion
      - B (torch.tensor): Has shape (num_of_diffeo, x_cutoff, y_cutoff), 
        parameterizes diffeo coefficient for y-distortion
    
    x_length by y_length matrix where num_of_terms elements is random and has flattened norm of 1
    in the case that num_of_terms << x_length * y_length, this matrix is sparse
    x_length and y_length specifies the frequency bound
    amplitude changes the norm
  '''
  if rng == None: rng = np.random.default_rng(seed = seed)

  A = []
  B = []

  total_num_elem = x_cutoff * y_cutoff
  pad_len = total_num_elem - num_of_terms #- 1

  for i in range(num_of_diffeo):

    A_sign = (rng.integers(2, size = num_of_terms) - 0.5) * 2
    B_sign = (rng.integers(2, size = num_of_terms) - 0.5) * 2

    if alpha == None:
      A_nm = diffeo_amp * rng.dirichlet(np.ones(num_of_terms)) * A_sign
      B_nm = diffeo_amp * rng.dirichlet(np.ones(num_of_terms)) * B_sign
    elif alpha != None:
      A_nm = diffeo_amp * rng.dirichlet(alpha * np.ones(num_of_terms)) * A_sign
      B_nm = diffeo_amp * rng.dirichlet(alpha * np.ones(num_of_terms)) * B_sign

    A_nm = np.pad(A_nm, (0, pad_len), 'constant')
    A_nm = rng.permutation(A_nm).reshape(x_cutoff, y_cutoff)
    # A_nm = np.pad(A_nm, (1,0), 'constant').reshape(x_length, y_length)
    A.append(t.Tensor(A_nm))

    B_nm = np.pad(B_nm, (0, pad_len), 'constant')
    B_nm = rng.permutation(B_nm).reshape(x_cutoff, y_cutoff)
    # B_nm = np.pad(B_nm, (1,0), 'constant').reshape(x_length, y_length)
    B.append(t.Tensor(B_nm))

  return t.stack(A), t.stack(B)



def create_grid_sample(x_length: int, 
                       y_length: int, 
                       A_list: torch.Tensor, 
                       B_list: torch.Tensor) -> torch.Tensor:
    '''
    Sin distortion for torch.nn.functional.grid_sample, the grid is from -1 to 1

    Args:
    - x_length (int): Length of x-axis of image.
    - y_length (int): Length of y-axis of image.
    - A_list (torch.Tensor): List of square matrices of coefficients, for x coordinate distortion
    - B_list (torch.Tensor): Same as A_list but for y coordinate distortion

    Returns:
    - torch.Tensor that has shape (N, x_length, y_length, 2) that can be fed into torch.nn.functional.grid_sample
    - the last dimension is length 2 because one is for x and one is for y.
    '''
    flow_grids = []

    for A_nm, B_nm in zip(A_list, B_list):
        non_zero_A_arg = torch.nonzero(A_nm, as_tuple=True)
        freq_A_arg = (torch.stack(non_zero_A_arg) + 1) * math.pi / 2
        max_A_freq = torch.max(torch.transpose(freq_A_arg, 1, 0), dim=1).values

        non_zero_B_arg = torch.nonzero(B_nm, as_tuple=True)
        freq_B_arg = (torch.stack(non_zero_B_arg) + 1) * math.pi / 2
        max_B_freq = torch.max(torch.transpose(freq_B_arg, 1, 0), dim=1).values

        unique_A_x, inv_index_A_x = torch.unique(freq_A_arg[1], return_inverse=True)
        unique_A_y, inv_index_A_y = torch.unique(freq_A_arg[0], return_inverse=True)
        unique_B_x, inv_index_B_x = torch.unique(freq_B_arg[1], return_inverse=True)
        unique_B_y, inv_index_B_y = torch.unique(freq_B_arg[0], return_inverse=True)

        x = torch.linspace(-1, 1, steps=x_length)
        y = torch.linspace(-1, 1, steps=y_length)
        X, Y = torch.meshgrid(x, y, indexing='ij')

        normalization_A = 1 / max_A_freq
        normalization_B = 1 / max_B_freq

        x_basis_A = torch.sin(unique_A_x[None, None, :] * (X[:, :, None] + 1))
        y_basis_A = torch.sin(unique_A_y[None, None, :] * (Y[:, :, None] + 1))
        x_basis_B = torch.sin(unique_B_x[None, None, :] * (X[:, :, None] + 1))
        y_basis_B = torch.sin(unique_B_y[None, None, :] * (Y[:, :, None] + 1))

        eps = torch.finfo(torch.float64).eps * 10
        x_basis_A[torch.abs(x_basis_A) < eps] = 0
        y_basis_A[torch.abs(y_basis_A) < eps] = 0
        x_basis_B[torch.abs(x_basis_B) < eps] = 0
        y_basis_B[torch.abs(y_basis_B) < eps] = 0


        X_pert = torch.einsum('xyi, i, xyi, i -> xy', y_basis_A[:, :, inv_index_A_y], A_nm[non_zero_A_arg], x_basis_A[:, :, inv_index_A_x], normalization_A)
        Y_pert = torch.einsum('xyi, i, xyi, i -> xy', y_basis_B[:, :, inv_index_B_y], B_nm[non_zero_B_arg], x_basis_B[:, :, inv_index_B_x], normalization_B)

        x_map = X + X_pert
        y_map = Y + Y_pert

        flow_grid_tensor = torch.stack((y_map, x_map), dim=-1)
        flow_grids.append(flow_grid_tensor.unsqueeze(0))

    return torch.cat(flow_grids, dim=0)


def get_id_grid(x_res, y_res, device = t.device('cpu')):
  '''
   Creates a grid of normalized coordinates from -1 to 1 in both x and y directions.
   
   Args:
       x_res: Resolution (number of points) in x direction
       y_res: Resolution (number of points) in y direction
       device: PyTorch device to place the output tensor on (default: CPU)
   
   Returns:
       id_grid: Tensor of shape [1, y_res, x_res, 2] containing normalized (Y,X) coordinates
   '''
  x = t.linspace(-1, 1, x_res)
  y = t.linspace(-1, 1, y_res)
  X, Y = t.meshgrid(x, y, indexing='ij')
  id_grid = t.cat([Y.unsqueeze(2), X.unsqueeze(2)], dim = 2).unsqueeze(0).to(device)
  return id_grid


#%%
def compose_diffeo_from_left(diffeo_l: t.tensor, diffeo_r: t.tensor, mode = 'bilinear', align_corners = True):
  '''
   Composes two diffeomorphisms by first applying r then l (l interpolates r).
   
   Args:
       diffeo_l: Left diffeomorphism tensor of shape (n, x_res, y_res, 2)
       diffeo_r: Right diffeomorphism tensor of shape (n, x_res, y_res, 2)
       mode: Interpolation mode for grid_sample
       align_corners: Grid sample alignment parameter
       
   Returns:
       product: Composed diffeomorphism tensor of shape (n, x_res, y_res, 2)
   '''
  if len(diffeo_l.shape) != 4 or len(diffeo_r.shape) != 4 or diffeo_l.shape[0] != diffeo_r.shape[0]:
    raise Exception(f'shape do not match, left:{diffeo_l.shape}, right:{diffeo_r.shape}')
  img = t.permute(diffeo_r, (0, 3, 1, 2))
  product = t.nn.functional.grid_sample(img, diffeo_l, mode = mode, padding_mode='border', align_corners= align_corners) # left multiplication
  product = t.permute(product, (0, 2, 3, 1))
  return product