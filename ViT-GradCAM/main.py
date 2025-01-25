import timm  # PyTorch image models library
import torch  # PyTorch deep learning library
from skimage import io  # Image processing from scikit-image
from torchvision.models import vgg19, resnet18  # Pretrained models from torchvision
from torchsummary import summary  # Model summary utility
from gradcam import GradCam, GradCamResNet  # Grad-CAM implementation
import numpy as np  # NumPy for numerical operations
import cv2  # OpenCV for image processing
from collections import OrderedDict
import sys
sys.path.append('/u/home/galc/VLP-Seminar/')
sys.path.append('/u/home/galc/VLP-Seminar/Finetune')
from train_cls import load_config
from methods.cls_model import FinetuneClassifier

# List all Vision Transformer models available in the timm library
timm.list_models('vit_*')

# Function to prepare input image for the model
def prepare_input(image):
    image = image.copy()  # Copy the image to avoid modifying the original

    # Normalize the image using the mean and standard deviation
    means = np.array([0.5, 0.5, 0.5])
    stds = np.array([0.5, 0.5, 0.5])
    image -= means
    image /= stds

    # Transpose the image to match the model's expected input format (C, H, W)
    image = np.ascontiguousarray(np.transpose(image, (2, 0, 1)))
    image = image[np.newaxis, ...]  # Add batch dimension

    return torch.tensor(image, requires_grad=True)  # Convert to PyTorch tensor

# Function to generate a Grad-CAM heatmap
def gen_cam(image, mask):
    # Create a heatmap from the Grad-CAM mask
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255

    # Superimpose the heatmap on the original image
    cam = (1 - 0.5) * heatmap + 0.5 * image
    cam = cam / np.max(cam)  # Normalize the result
    return np.uint8(255 * cam)  # Convert to 8-bit image

if __name__ == '__main__':
    # Load and preprocess the input image
    img = io.imread("both.png")
    img = np.float32(cv2.resize(img, (224, 224))) / 255  # Resize and normalize
    if img.shape[-1] == 4:  # Check for RGBA
        img = img[:, :, :3]
    inputs = prepare_input(img)  # Prepare the image for the model

    # Create the Vision Transformer model with pretrained weights
    # model = timm.create_model('vit_base_patch16_224', pretrained=True)
    # target_layer = model.blocks[-1].norm1  # Specify the target layer for Grad-CAM

    config = load_config('../configs/rsna.yaml')
    checkpoint_path = config['cls']['finetuned_checkpoint']
    checkpoint = torch.load(checkpoint_path)
    model = FinetuneClassifier(config)
    model_state_dict = model.state_dict()

    common_keys = set(checkpoint['state_dict'].keys()).intersection(set(model_state_dict.keys()))
    print(f"Number of common keys between checkpoint and model: {len(common_keys)}")
    
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    if config['cls']['backbone'] == 'vit_base':
        target_layer = model.img_encoder_q.model.blocks[-1].norm1
        grad_cam = GradCam(model, target_layer)
    else:
        target_layer = model.img_encoder_q.model.layer4[-1].conv3
        grad_cam = GradCamResNet(model, target_layer)
    
    # Initialize Grad-CAM with the model and target layer
    # grad_cam = GradCam(model, target_layer)
    mask = grad_cam(inputs)  # Compute the Grad-CAM mask
    result = gen_cam(img, mask)  # Generate the Grad-CAM heatmap

    # Save the result to an image file
    cv2.imwrite('result.jpg', result)
