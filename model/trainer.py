#%%
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision as tv
from torchvision.transforms import v2

from utils.diffeo_container import sparse_diffeo_container, DiffeoConfig
from utils.get_model_activation import retrieve_layer_activation
from utils.inverse_diffeo import find_param_inverse
from model.Unet.Unet import UNet, UnetConfig
from model.EDSR.EDSR import SingleScaleEDSR, SingleScaleEDSRConfig

import os
import logging
from dataclasses import dataclass, field, asdict
from typing import Tuple, Callable, List, Dict, Any


@dataclass
class TrainingConfig:
    """Configuration for training parameters"""
    optimizer: Callable = lambda param: optim.AdamW(
            param,
            lr=0.001,
            weight_decay=0.01
        )
    loss: Callable = nn.MSELoss()
    epochs: int = 40
    batch_size: int = 200
    num_training_images_per_class: int = 5
    num_channels_use_in_training: int = 8
    activation_layer: int = 5
    diffeo_sample_num: int = 8
    model_save_dir: str = field(default_factory=lambda: '/vast/xj2173/diffeo/scratch_data/Unet_weights/')
    brightness_penalty: int = 1

    def _extract_optimizer_info(self, optimizer_func: Callable) -> tuple[str, Dict[str, Any]]:
        """Extract optimizer name and parameters from a lambda or function."""
        # Create a dummy parameter to inspect the optimizer
        dummy_param = torch.nn.Parameter(torch.empty(1))
        optimizer_instance = optimizer_func([dummy_param])
        
        # Get optimizer class name
        optimizer_name = optimizer_instance.__class__.__name__
        
        # Extract parameters from the optimizer
        # Note: defaults is a dict containing all parameters including defaults
        defaults = optimizer_instance.defaults
        
        # Get the actual set parameters (not including unset defaults)
        param_groups = optimizer_instance.param_groups[0]
        actual_params = {
            key: value for key, value in param_groups.items()
            if key in defaults and key != 'params'  # exclude the params key
        }
        
        return optimizer_name, actual_params

    def __getstate__(self):
        state = asdict(self)  # Convert dataclass to dict
        
        # Extract optimizer info
        optimizer_name, optimizer_params = self._extract_optimizer_info(self.optimizer)
        
        # Replace optimizer function with serializable info
        state['optimizer'] = {
            'name': optimizer_name,
            'params': optimizer_params
        }
        return state

    def __setstate__(self, state):
        # Create mapping of optimizer names to classes
        optimizer_map = {
            'SGD': optim.SGD,
            'Adam': optim.Adam,
            'AdamW': optim.AdamW
        }
        
        # Get optimizer info
        optimizer_info = state['optimizer']
        optimizer_class = optimizer_map[optimizer_info['name']]
        optimizer_params = optimizer_info['params']
        
        # Reconstruct optimizer function
        state['optimizer'] = lambda params: optimizer_class(params, **optimizer_params)
        
        # Update instance
        self.__dict__.update(state)


#%%
class DiffeoDenoiseTrainer:
    def __init__(self, 
                 imagenet_path: str,
                 denoiser_model: str,
                 denoiser_config: dataclass,
                 train_config: TrainingConfig = None,
                 diffeo_config: DiffeoConfig = None,
                 device: torch.device = None):
        """
        Initialize the Diffeomorphism Denoising Trainer.

        Args:
            imagenet_path: Path to ImageNet dataset
            train_config: Training configuration
            diffeo_config: Diffeomorphism configuration
            denoiser_config: UNet denoiser architecture configuration
            device: Torch device to use for training
        """
        self.denoiser_model = denoiser_model
        self.denoiser_config = denoiser_config
        self.config = train_config or TrainingConfig()
        self.diffeo_config = diffeo_config or DiffeoConfig()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.imagenet_path = imagenet_path
        self.epoch_trained = 0
        self.train_losses = []
        self.test_losses = []
        self._setup_logging()
        self._initialize_components()

    def _setup_logging(self):
        job_id = os.getenv('SLURM_JOB_ID')
        logging.basicConfig(
            filename=f'slurm_{job_id}.err',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        logging.info('Logger initialized')
        logging.info(f"Training initialized on device: {self.device}")

    def _initialize_components(self):
        """Initialize model, dataset, and other components"""
        self.vision_model = self._setup_vision_model()
        logging.info('vision model initialized')
        self.dataset, self.test_dataset = self._setup_dataset()
        self.dataloader = self._setup_dataloader()
        logging.info('dataset and dataloader initialized')
        logging.info(f'setting up diffeo with {self.diffeo_config}')
        self.diffeos, self.inv_diffeos = self._setup_diffeos()
        logging.info('diffeo set up finished')
        self.denoiser = self._setup_denoiser()
        logging.info(f'denoiser initialized, has {self.param_count} number of parameters')
        logging.info(f'denoiser has {self.denoiser_config}')
        self.optimizer, self.loss_fn = self._setup_optimizer_loss()

    def _setup_vision_model(self, model_name: str = "efficientnet_v2_s"):
        """
        Sets up a pretrained vision model and its corresponding transform.
        Automatically handles model initialization and transform setup.
        Disables gradient computation for the vision model as it's used only for inference.
        
        Args:
            model_name: Name of the pretrained model to use
                       (default: "efficientnet_v2_s")
        
        Returns:
            tuple: (model, transform)
        """
        # Map of supported models to their weight enums
        model_config = {
            "efficientnet_v2_s": (
                tv.models.efficientnet_v2_s,
                tv.models.EfficientNet_V2_S_Weights.DEFAULT,
                tv.models.EfficientNet_V2_S_Weights.IMAGENET1K_V1
            ),
        }
        
        if model_name not in model_config:
            raise ValueError(f"Model {model_name} not supported. Choose from: {list(model_config.keys())}")
            
        model_fn, weights, transform_weights = model_config[model_name]
        
        # Initialize model
        model = model_fn(weights=weights).to(self.device)
        model.eval()
        
        # Disable gradient computation for the entire model
        for param in model.parameters():
            param.requires_grad = False
        
        # Setup transform
        self.transform = v2.Compose([
            lambda x: x.convert('RGB'),
            transform_weights.transforms()
        ])
        
        return model

    def _setup_dataset(self) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
        dataset = tv.datasets.ImageNet(
            self.imagenet_path, 
            split='train', 
            transform=self.transform
        )
        return self._get_dataset_subset(
            dataset, 
            self.config.num_training_images_per_class
        )

    def _setup_dataloader(self):
        return {
            'train': torch.utils.data.DataLoader(
                self.dataset, 
                batch_size=self.config.batch_size, 
                shuffle=True
            ),
            'test': torch.utils.data.DataLoader(
                self.test_dataset, 
                batch_size=self.config.batch_size, 
                shuffle=True
            )
        }

    def _setup_diffeos(self):
        diffeos = self._get_diffeo_container()
        inv_diffeos, _, _ = self._get_inv_diffeo_container(diffeos)
        return diffeos, inv_diffeos

    def _setup_denoiser(self, config: dataclass = None):
        """
        Sets up the UNet denoiser model.
        
        Args:
            config: UnetConfig object with model architecture parameters.
                   If None, uses default configuration.
        
        Returns:
            UNet: Configured denoiser model on the appropriate device
        """
        config = config or self.denoiser_config
        if self.denoiser_model == 'UNet':    
            denoiser = torch.compile(UNet(config.__dict__))
        if self.denoiser_model == 'SingleScaleEDSR':
            denoiser = torch.compile(SingleScaleEDSR(config))
        self.param_count = sum(p.numel() for p in denoiser.parameters())
        return (denoiser).to(self.device)

    def _setup_optimizer_loss(self):
        return self.config.optimizer(self.denoiser.parameters()), self.config.loss


    @staticmethod
    def _get_dataset_subset(dataset, num_image_per_class):
        class_ranges = []
        prev_class = 0
        start_idx = 0
        
        for i, target in enumerate(dataset.targets):
            if target != prev_class:
                class_ranges.append([start_idx, i])
                start_idx = i
                prev_class = target
        class_ranges.append([start_idx, i])

        train_indices = []
        test_indices = []
        for start, end in class_ranges:
            train_indices.extend(range(start, start + num_image_per_class))
            test_indices.append(start + num_image_per_class + 1)

        return (
            torch.utils.data.Subset(dataset, train_indices),
            torch.utils.data.Subset(dataset, test_indices)
        )

    @staticmethod
    def log_info(info: str):
        logging.info(info)

    def _get_diffeo_container(self) -> sparse_diffeo_container:
        dc = sparse_diffeo_container(
            self.diffeo_config.resolution, 
            self.diffeo_config.resolution
        )
        for strength in self.diffeo_config.strength:
            dc.sparse_AB_append(
                self.diffeo_config.x_range,
                self.diffeo_config.y_range,
                self.diffeo_config.num_nonzero_params,
                strength,
                self.diffeo_config.num_diffeo_per_strength
            )
        dc.get_all_grid()
        dc.to(self.device)
        return dc

    def _get_inv_diffeo_container(self, 
                                diffeo_container: sparse_diffeo_container, 
                                **kwargs) -> Tuple[sparse_diffeo_container, List, List]:
        AB = torch.stack([
            diffeo_container.A[0],
            diffeo_container.B[0]
        ])
        [A_inv, B_inv], loss_hist, mag_hist = find_param_inverse(
            AB, 
            resolution=128,
            device=self.device,
            **kwargs
        )
        
        inv_diffeos = sparse_diffeo_container(
            self.diffeo_config.resolution,
            self.diffeo_config.resolution,
            A=[A_inv.cpu()],
            B=[B_inv.cpu()]
        )
        inv_diffeos.get_all_grid()
        inv_diffeos.to(self.device)
        
        return inv_diffeos, loss_hist, mag_hist

    def get_diffeo_blurred_image(self, images):
        indices = torch.randperm(len(self.diffeos[0]))[:self.config.diffeo_sample_num]
        blurred_images = []
        
        for image in images:
            distorted = nn.functional.grid_sample(
                image.expand(self.config.diffeo_sample_num, -1, -1, -1),
                self.diffeos[0][indices],
                align_corners=True
            )
            blurred = nn.functional.grid_sample(
                distorted,
                self.inv_diffeos[0][indices],
                align_corners=True
            )
            blurred_images.append(blurred)
            
        return torch.stack(blurred_images)

    def _process_batch(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        images = images.to(self.device)
        activations, _ = retrieve_layer_activation(
            self.vision_model, 
            images, 
            [self.config.activation_layer]
        )
        activations = activations[self.config.activation_layer]
        indices = torch.randperm(activations.shape[1])[:self.config.num_channels_use_in_training]
        return activations[:, indices], self.get_diffeo_blurred_image(activations[:, indices])

    def train_epoch(self) -> Tuple[float, float]:
        self.denoiser.train()
        self.denoiser = self.denoiser
        train_loss = 0
        test_loss = 0
        
        # Training loop
        for images, _ in self.dataloader['train']:
            train_img, blurred = self._process_batch(images)
            for target, blurry in zip(train_img, blurred):
                self.optimizer.zero_grad()
                target = target.expand(len(blurry), -1, -1, -1)
                diff, channel, x, y = target.shape
                target = torch.reshape(target, (diff * channel, 1, x, y))
                blurry = torch.reshape(blurry, (diff * channel, 1, x, y))
                
                output = self.denoiser(blurry)
                brightness_constraint = torch.abs((output.mean([2,3]) - target.mean([2,3])))
                loss = self.loss_fn(output, target) + torch.mean((brightness_constraint + 1)**2 - 1) * self.config.brightness_penalty
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()

        # Validation loop
        self.denoiser.eval()
        with torch.no_grad():
            for images, _ in self.dataloader['test']:
                test_img, blurred = self._process_batch(images)
                
                for target, blurry in zip(test_img, blurred):
                    target = target.expand(len(blurry), -1, -1, -1)
                    diff, channel, x, y = target.shape
                    target = torch.reshape(target, (diff * channel, 1, x, y))
                    blurry = torch.reshape(blurry, (diff * channel, 1, x, y))
                    
                    output = self.denoiser(blurry)
                    test_loss += nn.MSELoss()(output, target).item()

        return (
            train_loss / (self.config.batch_size * len(self.dataloader['train'])),
            test_loss / (self.config.batch_size * len(self.dataloader['test']))
        )

    def train(self, save_dir: str):
        logging.info(f'training starts with {self.config.__getstate__()}')
        os.makedirs(save_dir, exist_ok=True)       

        for epoch in range(self.config.epochs):
            train_loss, test_loss = self.train_epoch()
            self.train_losses.append(train_loss)
            self.test_losses.append(test_loss)
            
            logging.info(f'Epoch {self.epoch_trained}: train_loss={train_loss:.6f}, test_loss={test_loss:.6f}')
            
            # Save model checkpoint
            torch.save({
                'training_config': self.config.__getstate__(),
                'denoiser_model': self.denoiser_model,
                'model_config': self.denoiser_config.__dict__,
                'diffeo_config': self.diffeo_config.__dict__,
                'model_state_dict': self.denoiser.state_dict(),
                'train_loss': self.train_losses,
                'test_loss': self.test_losses,
                'epoch_trained' : self.epoch_trained
            }, os.path.join(save_dir, f'model_weights_{self.epoch_trained}.pth'))

            self.epoch_trained += 1

def create_experiment_dir(base_dir: str, param_count: int) -> str:
    """Create a directory for the experiment based on model parameters"""
    return os.path.join(base_dir, f'{int(round(param_count/1000))}k/')