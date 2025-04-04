# 🚀 VLP finetune code for AI for Vision-Language Models in Medical Imaging (IN2107)

**This repository contains finetune code for the seminar "AI for Vision-Language Models in Medical Imaging (IN2107)." For more information, visit the [VLP Seminar page](https://compai-lab.github.io/teaching/vlm_seminar/).**

NOTE: This repository was modified to use both [MedCLIP](https://arxiv.org/pdf/2210.10163) and [ConVIRT](https://arxiv.org/pdf/2010.00747) method and finetune on RSNA and CheXpert5x200 classification task. 

The main code is designed to fine-tune Vision-Language Pre-trained models for downstream tasks, including classification, segmentation, and detection.

This project is built upon the code from [MGCA](https://github.com/HKU-MedAI/MGCA). A special thanks to their repository.

# 🗂️ Structure of the Repository
Here are the base structures of our repository (including the modifications for MedCLIP and ConVIRT): 
```
.
├── annotations # Stores the outputs of the preprocessing and annotations for each dataset (including the balanced RSNA dataset).
├── configs # Configuration files for each dataset (e.g., chexpert.yaml, rsna.yaml).
├── data # Outputs for the model (checkpoints, log outputs).
├── Finetune # Main code for fine-tuning the models and post-processing of results.
├── MedCLIP # GitHub Reposity of MedCLIP modified to use ViT backbone on classification finetune task.
├── poster # Final seminar poster with results.
├── preprocess_datasets # Code to preprocess the downstream datasets.
├── ViT-GradCAM # Code to plot saliency maps.
└── README.md
```
You can find the final seminar poster with our results in the [`poster`](poster) folder.

In [`postprocess.ipynb`](Finetune/postprocess/postprocess.ipynb) you can find our code to compute the confusion matrix of the test results on the RSNA dataset and to plot image embeddings for the CheXpert and RSNA datasets using t-SNE.

Both MedCLIP with ResNet50 and ConVIRT with the base Vision Transformer (ViT) can be run from the classification finetuning script [`train_cls.py`](Finetune/train_cls.py). Switching between the balanced and imbalanced versions of the RSNA dataset is also possible. Both RSNA balanced and imbalanced preprocessed datasets can be found in `annotations/rsna`. Details about the dataset balancing process can be found in the data preprocessing script [`rsna.ipynb`](preprocess_datasets/rsna.ipynb). 

We downloaded the pretrained checkpoints for MedCLIP directly from the official [`MedCLIP repository`](https://github.com/RyanWangZf/MedCLIP/tree/main/medclip). Our implementation for finetuning MedCLIP on RSNA to perform classification can be found in the [`MedCLIP`](MedCLIP) directory. 

For the ConVIRT ViT pretrained checkpoints please contact me directly.

# 🛠️ Preprocess Datasets
Here we provide examples for two datasets: RSNA and Chexpert.

## RSNA Dataset
The RSNA dataset includes:
- **Annotations**: Image, bounding box, and label.
- **Use Cases**: Suitable for tasks such as classification, detection, and segmentation (using bounding boxes as masks).

### Download the RSNA Dataset
To download the dataset, follow the MGCA setup, download via link [RSNA Pneumonia Detection Challenge](https://www.kaggle.com/competitions/rsna-pneumonia-detection-challenge/data):

```bash
mkdir ~/datasets/rsna
cd ~/datasets/rsna
kaggle competitions download -c rsna-pneumonia-detection-challenge
unzip rsna-pneumonia-detection-challenge.zip -d ./
```

### Preprocess the Dataset Format
Details can be found in [`rsna.ipynb`](preprocess_datasets/rsna.ipynb) under the `preprocess_datasets` folder. The outputs will be saved in `annotations/rsna/`:
- `train.csv`
- `val.csv`
- `test.csv`

## Chexpert Dataset
The Chexpert dataset includes:
- **Annotations**: Covers 14 diseases, with labels of 0, 1, and -1 (where 0 indicates absence, 1 indicates presence, and -1 indicates uncertainty).
- **Use Cases**: Primarily for classification tasks.

### Download the Chexpert Dataset
You can download the dataset from Kaggle using the following link: [Chexpert Dataset](https://www.kaggle.com/datasets/ashery/chexpert). Alternatively, you can download it directly via command line:

```bash
mkdir ~/datasets/chexpert
cd ~/datasets/chexpert
kaggle datasets download ashery/chexpert
unzip chexpert-v10-small.zip -d ./
```

# 🔧 Finetune
Note: The training part of the code is based on [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/starter/introduction.html).

## Code Structure for Finetune
The code structure for fine-tuning is divided into two main parts:
- **datasets**: Main part for loading different datasets.
- **methods**: Main part for different methods.
```
.Finetune
├── datasets # Main part for loading different datasets.
│   ├── cls_dataset.py
│   ├── data_module.py
│   ├── det_dataset.py
│   ├── __init__.py
│   ├── seg_dataset.py
│   ├── transforms.py
│   └── utils.py
|....
```
```
.Finetune
├── methods # Main parts for different methods.
│   ├── backbones # Contains the backbones needed for methods.
│   │   ├── cnn_backbones.py
│   │   ├── detector_backbone.py
│   │   ├── encoder.py
│   │   ├── __init__.py
│   │   ├── med.py
│   │   ├── transformer_seg.py
│   │   └── vits.py
│   ├── cls_model.py  # Model for classification.
│   ├── det_model.py # Model for detection.
│   ├── __init__.py
│   ├── seg_model.py # Model for segmentation.
│   └── utils # Some losses and utilities.
│       ├── box_ops.py
│       ├── detection_utils.py
│       ├── __init__.py
│       ├── segmentation_loss.py
│       └── yolo_loss.py
|....
```

## Example for Fine-tuning
Here we provide an example based on the RSNA dataset. You just need to modify the [rsna.yaml](configs/rsna.yaml) file for different tasks.

The `rsna.yaml` file contains four parts:
- **dataset**: Base information for the dataset.
- **cls**: Configuration for classification.
- **det**: Configuration for detection.
- **seg**: Configuration for segmentation.
```
dataset: # Base information for setting the dataset.
    img_size: 224 # The input size will be 224.
    dataset_dir: /u/home/lj0/datasets/RSNA_Pneumonia # Dataset base directory.
    train_csv: /u/home/lj0/Code/VLP-Seminars/annotations/rsna/train.csv # Annotations path for train, test, val.
    valid_csv: /u/home/lj0/Code/VLP-Seminars/annotations/rsna/val.csv
    test_csv: /u/home/lj0/Code/VLP-Seminars/annotations/rsna/test.csv
    
cls:
    img_size: 224
    backbone: resnet_50 # ResNet and ViT are supported (backbone you want to test).
    multilabel: False # Whether the classification task is a multilabel task.
    embed_dim: 128 
    in_features: 2048 
    num_classes: 2 # Number of classification classes.
    pretrained: True # Whether to utilize a pre-trained model to initialize.
    freeze: True # Whether to freeze the entire backbone.
    checkpoint: /home/june/Code/MGCA-main/data/ckpts/resnet_50.ckpt # Initialization checkpoint path.
    lr: 5.0e-4 # Learning rate for classification.
    dropout: 0.0 # Dropout rate.
    weight_decay: 1.0e-6 # Weight decay.

det:
    img_size: 224
    backbone: resnet_50 # Only ResNet-50 is supported.
    lr: 5.0e-4
    weight_decay: 1.0e-6
    conf_thres: 0.5 # Confidence threshold.
    iou_thres: [0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75] # IoU thresholds for YOLO.
    nms_thres: 0.5
    pretrained: True
    freeze: True
    max_objects:
    backbone: resnet_50 #only resnet_50 is supported
    lr: 5.0e-4
    weight_decay: 1.0e-6
    conf_thres: 0.5 # confidence thereshold 0.5
    iou_thres: [0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75] # iou thres for yolo vit_base # ViT-Base and ResNet-50.
    lr: 2e-4
    weight_decay: 0.05
    pretrained: True
    freeze: True
    embed_dim: 128  
    checkpoint: /home/june/Code/MGCA
seg:
    img_size: 224 #224 for vit
    backbone: vit_base #vit_base and resnet_50
    lr: 2e-4
    weight_decay: 0.05
    pretrained: True
    freeze: True
    embed_dim: 128  
    checkpoint: /home/june/Code/MGCA-main/data/ckpts/vit_base.ckpt

```

### 1) Classification (Finetune)
You can directly observe the performance of classification using Weights & Biases (WandB).
```bash
python train_cls.py --batch_size 46 --num_workers 16 --max_epochs 50 --config ../configs/chexpert.yaml --gpus 1 --dataset chexpert --data_pct 1 --ckpt_dir ../data/ckpts --log_dir ../data/log_output
```
Alternatively, you can directly run:
```bash
python train_cls.py
```

These experiments were run with MedCLIP using Resnet 50 backbone and ConVIRT using ViT backbone. 
With these experiments we tried to reveal the role of balancing datasets and of using different backbone architectures.


### 2) Detection (Finetune)
You can directly observe the performance of classification using Weights & Biases (WandB).
```bash
python train_det.py --batch_size 32 --num_workers 16 --max_epochs 50 --config ../configs/rsna.yaml --gpus 1 --dataset rsna --data_pct 1 --ckpt_dir ../data/ckpts --log_dir ../data/log_output
```
Alternatively, you can directly run:
```bash
python train_det.py
```

### 3) Segmentation (Finetune)
You can directly observe the performance of classification using Weights & Biases (WandB).
```bash
python train_seg.py --batch_size 48 --num_workers 4 --max_epochs 50 --config ../configs/rsna.yaml --gpus 1 --dataset rsna --data_pct 1 --ckpt_dir ../data/ckpts --log_dir ../data/log_output
```
Alternatively, you can directly run:
```bash
python train_seg.py
```

# 🎉 End & Enjoy
