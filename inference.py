import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import numpy as np

import tensorflow as tf
from tensorflow_examples.models.pix2pix import pix2pix

import torch
from torchvision import transforms
from segmentation_models_pytorch import Unet

import cv2

from PIL import Image

from cycle_GAN import LinearDecayLR
from utils import load_images, save_generated_images, CombineImages, CreateVideoFromImages


OUTPUT_CHANNELS = 3

# Initial learning rate, final learning rate, and epochs settings
EPOCHS = 200
total_epochs = EPOCHS  # Assuming EPOCHS is defined as 200
initial_lr = 2e-4
final_lr = 2e-6
decay_start_epoch = 100

# Defining the generators and the discriminators
# Defining the Pix2Pix Generators
# generator_g = pix2pix.unet_generator(OUTPUT_CHANNELS, norm_type='instancenorm')
generator_f = pix2pix.unet_generator(OUTPUT_CHANNELS, norm_type='instancenorm')

# # Defining the Pix2Pix Discriminators
# discriminator_x = pix2pix.discriminator(norm_type='instancenorm', target=False)
# discriminator_y = pix2pix.discriminator(norm_type='instancenorm', target=False)

# Defining the optimisers for the generators and the discriminators
# For the generators
# generator_g_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
generator_f_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

# # # For the discriminators
# discriminator_x_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
# discriminator_y_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

optimizers = [
    # generator_g_optimizer,
    generator_f_optimizer,
    # discriminator_x_optimizer,
    # discriminator_y_optimizer
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

print('Loading CycleGAN model...')
ckpt = tf.train.Checkpoint(
    # generator_g=generator_g,
    generator_f=generator_f,
    # discriminator_x=discriminator_x,
    # discriminator_y=discriminator_y,
    # generator_g_optimizer=generator_g_optimizer,
    generator_f_optimizer=generator_f_optimizer,
    # discriminator_x_optimizer=discriminator_x_optimizer,
    # discriminator_y_optimizer=discriminator_y_optimizer
    )

checkpoint_path = 'checkpoints/train'
ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

# If a checkpoint exists, restore the latest checkpoint.

if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print ('Latest checkpoint restored')
else:
    print('No checkpoint found')
    exit()

print('Loading Segmentation model...')
#Load the trained model
model = Unet(
    encoder_name="resnet34",  #Choose the same encoder used during training
    encoder_weights=None,  #No pre-trained weights since we're loading our custom-trained model
    in_channels=3,
    classes=1
)
model.load_state_dict(torch.load("segmentation_model/person_segmentation_unet_final.pth", map_location=torch.device('cpu')))
# model = model.cuda()
model.eval()

#Directories to save foreground, background, and mask images
fg_dir = 'segmentation_outputs/fg'
bg_dir = 'segmentation_outputs/bg'
mask_dir = 'segmentation_outputs/mask'  #Directory for saving masks
os.makedirs(fg_dir, exist_ok=True)
os.makedirs(bg_dir, exist_ok=True)
os.makedirs(mask_dir, exist_ok=True)

#Transformation to be applied on each frame

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

#Load the video from a folder
cap = cv2.VideoCapture('test_videos/test_video.mp4')
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

frame_idx = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    #Convert frame to RGB (cv2 reads in BGR by default)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    #Apply transformations
    input_image = Image.fromarray(frame_rgb)
    input_tensor = transform(input_image).unsqueeze(0)

    #Perform segmentation
    with torch.no_grad():
        output = model(input_tensor)
        output_mask = torch.sigmoid(output).squeeze().cpu().numpy()
    
    #Resize the mask back to original frame size
    original_size = (frame.shape[1], frame.shape[0])  # (width, height)
    output_mask = cv2.resize(output_mask, original_size)

    #Threshold the mask to create binary mask
    binary_mask = (output_mask > 0.5).astype(np.uint8)

    #Create the foreground image
    fg_image = cv2.bitwise_and(frame, frame, mask=binary_mask)

    #Invert the mask to create a background mask
    inv_mask = cv2.bitwise_not(binary_mask * 255)

    #Convert the inverted mask to 3 channels
    inv_mask_3c = cv2.cvtColor(inv_mask, cv2.COLOR_GRAY2BGR)

    #Create the background image by removing the foreground
    bg_image = cv2.bitwise_and(frame, inv_mask_3c)

    #Save the foreground, background, and mask images
    fg_path = os.path.join(fg_dir, f'{frame_idx:04d}_f.png')
    bg_path = os.path.join(bg_dir, f'{frame_idx:04d}_b.png')
    mask_path = os.path.join(mask_dir, f'{frame_idx:04d}_m.png')  #Mask path

    cv2.imwrite(fg_path, fg_image)
    cv2.imwrite(bg_path, bg_image)
    
    #Save the mask as a black and white image (white = 255 for foreground, black = 0 for background)
    cv2.imwrite(mask_path, binary_mask * 255)

    #Move to the next frame
    frame_idx += 1
    print(f"Processed frame {frame_idx}/{frame_count}")

cap.release()
print("Segmentation process complete...")


# real_images = 'extracted_frames'
# good_images = load_images(real_images)

# Applying image-to-image translation to fg and bg
fg_bg_images = [fg_dir, bg_dir]

for image_section in fg_bg_images:
    images = load_images(image_section)

    if image_section == 'segmentation_outputs/fg':
        folder_name = 'fg'
    else:
        folder_name = 'bg'

    output_dir = f'output_images/generated_{folder_name}'

    save_generated_images(generator_f, images, output_dir=output_dir)

    # image_folder = output_dir  # Folder where your images are saved
    # video_name = 'video_name.mp4'  # Name for the output video file
    # fps = 24 # Frames per second (adjust based on your need)

    # images_to_video(image_folder, video_name, fps)

combined_dir = 'segmentation_outputs/combined'  #Directory to save combined images
os.makedirs(combined_dir, exist_ok=True)

#Get the list of background, foreground, and mask images
bg_images = sorted(os.listdir(bg_dir))
fg_images = sorted(os.listdir(fg_dir))
mask_images = sorted(os.listdir(mask_dir))

#Iterate over the images and combine them
for idx in range(len(bg_images)):
    CombineImages(
        idx=idx, 
        bg_dir=bg_dir, 
        fg_dir=fg_dir, 
        mask_dir=mask_dir, 
        combined_dir=combined_dir,
        bg_images=bg_images,
        fg_images=fg_images,
        mask_images=mask_images
        )

fps = 24
CreateVideoFromImages(combined_dir, fps)