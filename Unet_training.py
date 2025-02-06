#%%
import torch
import torch.optim as optim
from model.trainer import DiffeoDenoiseTrainer
from model.trainer import TrainingConfig, DiffeoConfig, create_experiment_dir
from model.Unet.Unet import UnetConfig
from model.EDSR.EDSR import SingleScaleEDSRConfig

# if __name__ == "__main__":
# Configuration
torch.backends.cudnn.benchmark = True

# model = 'UNet'
# denoiser_config = UnetConfig(
#     initial_filters = 48, 
#     depth=0, 
#     num_resid_blocks=[20],
#     num_conv_in_resid=[2],
#     num_conv_in_skip=[],
#     conv_kwargs={'kernel_size':3, 'padding':1},
#     upsample_kwargs={'mode':'bilinear', 'align_corners': True})

model = 'SingleScaleEDSR'
denoiser_config = SingleScaleEDSRConfig(
    20,
    [48,96],
    [96,48],
    [3,3]
)
train_config = TrainingConfig(
    optimizer = lambda param: optim.AdamW(
            param,
            lr=0.001,
            weight_decay=0.01
        ), 
    loss = torch.nn.MSELoss(),
    batch_size= 500,
    epochs = 10, num_training_images_per_class= 10, num_channels_use_in_training= 24, diffeo_sample_num=1,
    activation_layer = 5)

diffeo_config = DiffeoConfig(
    resolution = 192,
    x_range = [0,5],
    y_range = [0,5],
    num_nonzero_params = 3,
    strength = [0.1],
    num_diffeo_per_strength = 300)
#%%
# Initialize trainer
trainer = DiffeoDenoiseTrainer(
    imagenet_path='/imagenet',
    denoiser_model = model,
    train_config=train_config,
    diffeo_config=diffeo_config,
    denoiser_config=denoiser_config
)
# Create experiment directory and start training
save_dir = create_experiment_dir(train_config.model_save_dir, trainer.param_count)
trainer.train(save_dir)

# Switch Optimizer
trainer.config.optimizer = lambda param: optim.SGD(
            param,
            lr=0.00015,
            weight_decay=0.0001,
            momentum = 0.8
        )
trainer.config.epochs = 10
trainer.optimizer, trainer.loss_fn = trainer._setup_optimizer_loss()
trainer.log_info(f'switched to SGD for {trainer.config.epochs} more epochs')
trainer.train(save_dir)

# Manually lowering learning rate
trainer.config.optimizer = lambda param: optim.SGD(
            param,
            lr=0.00001,
            weight_decay=0.00001,
            momentum = 0.8
        )
trainer.config.epochs = 20
trainer.optimizer, trainer.loss_fn = trainer._setup_optimizer_loss()
trainer.log_info(f'Lowered SGD LR for {trainer.config.epochs} more epochs')
trainer.train(save_dir)