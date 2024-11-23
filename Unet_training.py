#%%
import torch
import torch.optim as optim
from model.Unet.trainer import DiffeoDenoiseTrainer
from model.Unet.trainer import DenoiserConfig, TrainingConfig, DiffeoConfig, create_experiment_dir

# if __name__ == "__main__":
# Configuration
torch.backends.cudnn.benchmark = True

denoiser_config = DenoiserConfig(
    initial_filters = 48, 
    depth=0, 
    num_resid_blocks=[25],
    num_conv_in_resid=[1],
    num_conv_in_skip=[],
    conv_kwargs={'kernel_size':5, 'padding':2},
    upsample_kwargs={'mode':'bilinear', 'align_corners': True})

train_config = TrainingConfig(
    optimizer = lambda param: optim.AdamW(
            param,
            lr=0.001,
            weight_decay=0.01
        ), 
    batch_size= 200,
    epochs = 10, num_training_images_per_class= 5, num_channels_use_in_training= 24, diffeo_sample_num=1,
    activation_layer = 5)

diffeo_config = DiffeoConfig(
    blurry_resolution = 192,
    x_range = [0,3],
    y_range = [0,3],
    num_nonzero_params = 3,
    strength = [0.1],
    total_random_diffeos = 200)
#%%
# Initialize trainer
trainer = DiffeoDenoiseTrainer(
    imagenet_path='/imagenet',
    train_config=train_config,
    diffeo_config=diffeo_config,
    denoiser_config=denoiser_config
)
# Create experiment directory and start training
save_dir = create_experiment_dir(train_config.model_save_dir, trainer.param_count)
trainer.train(save_dir)
trainer.config.optimizer = lambda param: optim.SGD(
            param,
            lr=0.0005,
            weight_decay=0.0001,
            momentum = 0.8
        )
trainer.config.epochs = 20
trainer.optimizer, trainer.loss_fn = trainer._setup_optimizer_loss()
trainer.log_info(f'switched to SGD with {trainer.config.optimizer} for {trainer.config.epochs} more epochs')
trainer.train(save_dir)



# %%
