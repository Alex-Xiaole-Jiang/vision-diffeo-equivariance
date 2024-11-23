#%%
from model.Unet.trainer import DiffeoDenoiseTrainer
from model.Unet.trainer import DenoiserConfig, TrainingConfig, DiffeoConfig, create_experiment_dir

# if __name__ == "__main__":
# Configuration
denoiser_config = DenoiserConfig(
    initial_filters = 32, 
    depth=1, 
    num_resid_blocks=[12,6], 
    num_conv_in_resid=[2,2],
    num_conv_in_skip=[2],
    conv_kwargs={'kernel_size':5, 'padding':2}, upsample_kwargs={'mode':'bilinear', 'align_corners': True})

train_config = TrainingConfig(
    learning_rate = 0.001, weight_decay = 0.01, batch_size= 200,
    epochs = 40, num_training_images_per_class= 5, num_channels_use_in_training= 8, diffeo_sample_num=8,
    activation_layer = 5)

diffeo_config = DiffeoConfig(
    blurry_resolution = 192,
    x_range = [0,3],
    y_range = [0,3],
    num_nonzero_params = 3,
    strength = [0.1],
    total_random_diffeos = 200)

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




# %%
