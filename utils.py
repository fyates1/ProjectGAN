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