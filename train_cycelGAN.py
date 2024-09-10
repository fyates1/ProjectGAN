import time

from tqdm import tqdm
from IPython.display import clear_output

import numpy as np

import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow_examples.models.pix2pix import pix2pix

from cycle_GAN import LinearDecayLR, train_step, generate_images
from utils import load_images, truncate_datasets


real_images = 'real2anime/real'
anime_images = 'real2anime/anime'
checkpoint_path = '/checkpoints/train/'

# Initial learning rate, final learning rate, and epochs settings
EPOCHS = 200
total_epochs = EPOCHS  # Assuming EPOCHS is defined as 200
initial_lr = 2e-4
final_lr = 2e-6
decay_start_epoch = 100

good_images = load_images(real_images)
bad_images = load_images(anime_images)

good_images, bad_images = truncate_datasets(good_images, bad_images)

# Displa samples of real images
fig,axs = plt.subplots(nrows=4,ncols=4,figsize=(8,8))
for i in range(16):
    ax = axs[i//4,i%4]
    sample = good_images[i]
    sample = (((sample * 127.5) + 127.5)).astype(np.uint8)
    ax.imshow((sample))
    ax.axis('off')
plt.show()

OUTPUT_CHANNELS = 3

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

ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

# If a checkpoint exists, restore the latest checkpoint.
if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print ('Latest checkpoint restored')

for epoch in range(EPOCHS):

    BATCH_SIZE=1 #According to all papers batch size must be 1 still defined it to maintain 4 dims when passed to model
    n = 0
    linear_decay_lr.on_epoch_begin(epoch)

    start = time.time()
    for i in tqdm(range(0, len(good_images), BATCH_SIZE)):
        image_x = bad_images[i:i+BATCH_SIZE]
        image_y = good_images[i:i+BATCH_SIZE]

        train_step(
            image_x=image_x, 
            image_y=image_y, 
            generator_g=generator_g, 
            generator_f=generator_f, 
            discriminator_x=discriminator_x, 
            discriminator_y=discriminator_y, 
            generator_g_optimizer=generator_g_optimizer, 
            generator_f_optimizer=generator_f_optimizer, 
            discriminator_x_optimizer=discriminator_x_optimizer, 
            discriminator_y_optimizer=discriminator_y_optimizer
            )
        
        if i % 100 == 0:
            generate_images(generator_f, image_y)
        clear_output(wait=True)

    end = time.time()
    elapsed_time = end - start
    # Using a consistent image so that the progress of the model is clearly visible
    for x in range(len(good_images[:15])):
        image_x = good_images[x:x+BATCH_SIZE]
        generate_images(generator_f, image_x)

    if (epoch + 1) % 1 == 0:
        ckpt_save_path = ckpt_manager.save()
        print (f'Saving checkpoint for epoch {epoch+1} at {ckpt_save_path}')

    print (f'Time taken for epoch {epoch + 1} is {elapsed_time} sec \n')
