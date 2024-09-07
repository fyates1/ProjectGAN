import os
import cv2
import numpy as np

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