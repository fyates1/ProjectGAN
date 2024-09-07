import tensorflow as tf
from tensorflow_examples.models.pix2pix import pix2pix

from cycle_GAN import LinearDecayLR, generate_images
from utils import load_images


OUTPUT_CHANNELS = 3

# Initial learning rate, final learning rate, and epochs settings
EPOCHS = 200
total_epochs = EPOCHS  # Assuming EPOCHS is defined as 200
initial_lr = 2e-4
final_lr = 2e-6
decay_start_epoch = 100

# Defining the generators and the discriminators
# Defining the Pix2Pix Generators
generator_g = pix2pix.unet_generator(OUTPUT_CHANNELS, norm_type='instancenorm')
generator_f = pix2pix.unet_generator(OUTPUT_CHANNELS, norm_type='instancenorm')

# Defining the Pix2Pix Discriminators
discriminator_x = pix2pix.discriminator(norm_type='instancenorm', target=False)
discriminator_y = pix2pix.discriminator(norm_type='instancenorm', target=False)

# Defining the optimisers for the generators and the discriminators
# For the generators
generator_g_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
generator_f_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

# For the discriminators
discriminator_x_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_y_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

optimizers = [
    generator_g_optimizer,
    generator_f_optimizer,
    discriminator_x_optimizer,
    discriminator_y_optimizer
]

linear_decay_lr = LinearDecayLR(
    initial_lr=initial_lr,
    final_lr=final_lr,
    decay_start_epoch=decay_start_epoch,
    total_epochs=total_epochs,
    optimizers=optimizers
)

linear_decay_lr = LinearDecayLR(
    initial_lr=initial_lr,
    final_lr=final_lr,
    decay_start_epoch=decay_start_epoch,
    total_epochs=total_epochs,
    optimizers=optimizers
)

ckpt = tf.train.Checkpoint(
    generator_g=generator_g,
    generator_f=generator_f,
    discriminator_x=discriminator_x,
    discriminator_y=discriminator_y,
    generator_g_optimizer=generator_g_optimizer,
    generator_f_optimizer=generator_f_optimizer,
    discriminator_x_optimizer=discriminator_x_optimizer,
    discriminator_y_optimizer=discriminator_y_optimizer
    )

checkpoint_path = '/checkpoints/train/'
ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

# If a checkpoint exists, restore the latest checkpoint.
if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print ('Latest checkpoint restored')
else:
    print('No checkpoint found')
    exit()

real_images = 'real2anime/real'
good_images = load_images(real_images)
generate_images(generator_f, good_images)