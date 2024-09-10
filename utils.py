import os
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import array_to_img
import tensorflow as tf

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

def save_generated_images(model, input_images, output_dir='output_images/3'):
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


def numerical_sort(value):
    """
    Helper function to sort filenames numerically.
    Extracts the numeric part from the file name for proper sorting.
    """
    return int(''.join(filter(str.isdigit, value)))


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
        image_path = os.path.join(image_folder, image)
        frame = cv2.imread(image_path)
        video.write(frame)
        print(f"Adding {image} to video")

    # Release the video writer object
    video.release()
    print(f"Video saved as {video_name}")
