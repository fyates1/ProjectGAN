import json

import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import supervisely as sly

from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from segmentation_models_pytorch import Unet

#Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class SuperviselySegmentationDataset(Dataset):
    """
    A custom dataset class that works with Supervisely data.
    """

    def __init__(self, project, transform=None):
        """
        Initialise the dataset with the project and transform.

        Parameters:
        project (sly.Project): The Supervisely project.
        transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.project = project
        self.transform = transform
        self.image_ann_pairs = []

        #Iterate through the datasets and collect image and annotation pairs
        for dataset in self.project.datasets:
            for item_name, image_path, ann_path in dataset.items():
                self.image_ann_pairs.append((image_path, ann_path))

    def __len__(self):
        """
        Return the total number of image-annotation pairs.
        """
        return len(self.image_ann_pairs)

    def __getitem__(self, idx):
        """
        Retrieve the image and mask corresponding to the index.

        Parameters:
        idx (int): Index of the sample.

        Returns:
        tuple: Transformed image and mask.
        """
        image_path, ann_path = self.image_ann_pairs[idx]
        img = sly.image.read(image_path)  #Read the image
        ann_json = json.load(open(ann_path))
        ann = sly.Annotation.from_json(ann_json, self.project.meta)

        #Create an empty mask
        mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)

        #Draw each label on the mask
        for label in ann.labels:
            label.draw(mask, color=1)

        mask = Image.fromarray(mask * 255)  #Convert to PIL Image for transformations

        if self.transform:
            img = self.transform(img)
            mask = self.transform(mask)

        return img, mask

#Set the input directory to your 'persons' dataset
input_dir = "./persons"

#Create a Supervisely project from the local directory
project = sly.Project(input_dir, sly.OpenMode.READ)
print("Opened project: ", project.name)
print("Number of images in project:", project.total_items)

#Show annotations, tags, and classes
print(project.meta)

#Iterate over classes in the project, showing their names, geometry types, and colours
for obj_class in project.meta.obj_classes:
    print(
        f"Class '{obj_class.name}': geometry='{obj_class.geometry_type}', colour='{obj_class.color}'",
    )

#Iterate over tags in the project, showing their names and colours
for tag in project.meta.tag_metas:
    print(f"Tag '{tag.name}': colour='{tag.color}'")

print("Number of datasets (aka folders) in project:", len(project.datasets))


#Define the transformation to be applied to the images and masks
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((256, 256))  #Resize to the same size as the U-Net input
])

#Create the dataset and dataloader
dataset = SuperviselySegmentationDataset(project, transform=transform)
dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

#Define the U-Net model
model = Unet(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=3,
    classes=1
)
model = model.to(device)  #Ensure the model is on the GPU

#Define the loss function and optimiser
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

#Early stopping parameters
patience = 5  #Number of epochs with no improvement after which training will be stopped
min_delta = 0.001  #Minimum change in the monitored quantity to qualify as an improvement
best_loss = np.inf
early_stop_counter = 0

#Training loop with tqdm for progress monitoring
num_epochs = 25
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    #Use tqdm to monitor the progress of the training loop
    with tqdm(total=len(dataloader), desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch") as pbar:
        for images, masks in dataloader:
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks.float())
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            #Update the tqdm bar
            pbar.set_postfix({"Loss": f"{running_loss/len(dataloader):.4f}"})
            pbar.update(1)

    epoch_loss = running_loss / len(dataloader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")

    #Early stopping logic
    if epoch_loss < best_loss - min_delta:
        best_loss = epoch_loss
        early_stop_counter = 0  #Reset the counter if we see an improvement
        torch.save(model.state_dict(), "person_segmentation_unet_best.pth")  #Save the best model
    else:
        early_stop_counter += 1

    if early_stop_counter >= patience:
        print("Early stopping triggered")
        break

#Save the final trained model
torch.save(model.state_dict(), "person_segmentation_unet_final.pth")
