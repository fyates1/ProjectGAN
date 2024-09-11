import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import array_to_img

def load_images(path):
    images = []
    for filename in os.listdir(path):
        img = cv2.imread(os.path.join(path,filename))
        if img is not None:
            img = cv2.resize(img, (256,256)) # resize the image again to the desired size
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = (img.astype('float32') - 127.5)/127.5 # normalize pixel values
            images.append(img)
    images = np.array(images)
    return images

def load_images_255(path):
    images = []
    for filename in os.listdir(path):
        img = cv2.imread(os.path.join(path,filename))
        if img is not None:
            img = cv2.resize(img, (256,256)) # resize the image again to the desired size
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images.append(img)
    images = np.array(images)
    return images



def truncate_datasets(dataset_x, dataset_y):
    min_size = min(len(dataset_x), len(dataset_y))

    return dataset_x[:min_size], dataset_y[:min_size]


def video_to_frames(video_path, output_folder='frames/', frame_prefix='frame_'):
    """
    Extracts frames from a video and saves them as images.
    
    Parameters:
    - video_path: Path to the input video file.
    - output_folder: Folder where the frames will be saved.
    - frame_prefix: Prefix for the frame filenames.
    """
    
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error: Couldn't open the video file.")
        return
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Save the frame as an image
        frame_filename = os.path.join(output_folder, f"{frame_prefix}{frame_count:04d}.png")
        cv2.imwrite(frame_filename, frame)
        print(f"Saved {frame_filename}")
        
        frame_count += 1
    
    # Release the video capture object
    cap.release()
    print("Frame extraction completed.")

def save_generated_images(model, input_images, output_dir='output_images/'):
    """
    Passes each image in the input_images list through the model and saves the generated output.
    
    Parameters:
    - model: The model (generator) used to generate images.
    - input_images: List of input images (assumed to be already preprocessed).
    - output_dir: The directory where the generated images will be saved.
    """
    
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for idx, image in enumerate(input_images):
        # Add a batch dimension to the image before passing it through the model
        image = tf.expand_dims(image, axis=0)
        
        # Generate the image using the model
        generated_image = model(image, training=True)
        
        # Remove batch dimension and convert the generated tensor to an image
        generated_image = array_to_img(generated_image[0])
        
        # Save the generated image with a unique name
        image_path = os.path.join(output_dir, f"{idx}.png")
        generated_image.save(image_path)
        print(f"Image saved: {image_path}")


def numerical_sort(values):
    sorted_values = []
    for value in values:
        try:
            num = int(''.join(filter(str.isdigit, value)))
            sorted_values.append((num, value))
        except ValueError:
            # Ignore files that can't be converted to a number
            continue
    # Sort by the extracted number and return only the file names
    sorted_values.sort(key=lambda x: x[0])
    return [val[1] for val in sorted_values]



def images_to_video(image_folder, video_name, fps=30):
    """
    Combines images from a folder into a video.
    
    Parameters:
    - image_folder: Folder containing images to combine into the video.
    - video_name: Name of the output video file (e.g., 'output_video.mp4').
    - fps: Frames per second for the video.
    """
    
    # Get all image file names from the folder and sort them
    images = [img for img in os.listdir(image_folder) if img.endswith((".png", ".jpg", ".jpeg"))]
    images.sort(key=numerical_sort)

    if len(images) == 0:
        print("No images found in the folder!")
        return

    # Read the first image to get dimensions
    first_image_path = os.path.join(image_folder, images[0])
    frame = cv2.imread(first_image_path)
    height, width, layers = frame.shape

    # Define video codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 files
    video = cv2.VideoWriter(video_name, fourcc, fps, (width, height))

    # Loop through each image, read it, and add it to the video
    for image in images:
        try:
            image_path = os.path.join(image_folder, image)
            frame = cv2.imread(image_path)
            video.write(frame)
            # print(f"Adding {image} to video")
        except:
            print(f'Issues with frame {frame}')
            continue

    # Release the video writer object
    video.release()
    print(f"Video saved as {video_name}")

def CombineImages(idx, bg_dir, fg_dir, mask_dir, combined_dir, bg_images, fg_images, mask_images):
    """
    Combine corresponding background, foreground, and mask images into a single composite image.

    Args:
        idx (int): The index of the image being processed.

    """
    

    #Load background, foreground, and mask images
    bg_image_path = os.path.join(bg_dir, bg_images[idx])
    fg_image_path = os.path.join(fg_dir, fg_images[idx])
    print(fg_image_path,bg_image_path)
    mask_image_path = os.path.join(mask_dir, mask_images[idx])

    bg_image = cv2.imread(bg_image_path)  #Background
    fg_image = cv2.imread(fg_image_path)  #Foreground
    mask_image = cv2.imread(mask_image_path, cv2.IMREAD_GRAYSCALE)  #Mask in grayscale

    #Ensure mask is binary (black and white) and corresponds to the foreground subject
    _, binary_mask = cv2.threshold(mask_image, 128, 255, cv2.THRESH_BINARY)

    #Resize mask and foreground to match the size of the background
    fg_image = cv2.resize(fg_image, (bg_image.shape[1], bg_image.shape[0]))
    binary_mask = cv2.resize(binary_mask, (bg_image.shape[1], bg_image.shape[0]))

    #Apply the mask to the foreground to remove any bleed over the edges
    fg_image_masked = cv2.bitwise_and(fg_image, fg_image, mask=binary_mask)

    #Create an inverse mask to mask out the area in the background where the foreground will be placed
    inv_mask = cv2.bitwise_not(binary_mask)

    #Mask the background where the foreground will be placed
    bg_image_masked = cv2.bitwise_and(bg_image, bg_image, mask=inv_mask)

    #Combine the masked background and the masked foreground
    combined_image = cv2.add(bg_image_masked, fg_image_masked)

    #Save the combined image
    combined_image_path = os.path.join(combined_dir, f'{idx:04d}_combined.png')
    cv2.imwrite(combined_image_path, combined_image)
    print(combined_image_path)
    print(f"Processed and saved: {combined_image_path}")

def CreateVideoFromImages(path, output_path, fps):
    """
    Creates an MP4 video from a sequence of images stored in the 'combined' folder.
    """
    #Directory containing combined images
    combined_dir = path

    #Output video file
    output_video = f'{output_path}/final_combined_output.mp4'

    #Get the list of image files in the combined folder
    image_files = sorted([f for f in os.listdir(combined_dir) if f.endswith('.png')])

    #Load the first image to get the width and height for the video
    first_image_path = os.path.join(combined_dir, image_files[0])
    frame = cv2.imread(first_image_path)
    height, width, layers = frame.shape

    #Define the video codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  #Codec for MP4
    video = cv2.VideoWriter(output_video, fourcc, fps, (width, height))  # 30 FPS

    #Loop through the images and write them to the video file
    for image_file in image_files:
        image_path = os.path.join(combined_dir, image_file)
        frame = cv2.imread(image_path)
        video.write(frame)  #Add frame to the video

    #Release the video writer
    video.release()

    print(f"Video saved as {output_video}")
