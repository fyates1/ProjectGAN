
# CycleGAN Project

## Overview

This project implements a CycleGAN model for image-to-image translation. CycleGAN is a type of Generative Adversarial Network (GAN) that can learn mappings between two domains without paired examples. The model is implemented using TensorFlow and PyTorch.

## Key Features

- **Generators and Discriminators**: Implements Pix2Pix-based generators and discriminators.
- **Loss Functions**: Includes adversarial, cycle-consistency, and identity loss functions.
- **Learning Rate Scheduling**: Implements a linear decay learning rate schedule.
- **Inference Pipeline**: Supports both image and video processing.


## Installation

To install the required dependencies, run:

```bash
pip install -r requirements.txt
```
Note that a python 3.10.14 conda environment will be required 
## Usage

### Training

To train the model, use:

```bash
python cycle_GAN.py
```

### Inference

To run inference on test images or videos:

```bash
python inference.py
```

## Results

The generated images will be saved in the `output_images` directory. To visualize the results, you can check the `segmentation_outputs` folder for segmented images and combined video frames.
