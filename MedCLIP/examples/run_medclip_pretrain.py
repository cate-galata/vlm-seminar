import pdb, os
import random

import warnings
warnings.filterwarnings("ignore")

from PIL import Image
from sklearn.utils import shuffle
from pydicom.errors import InvalidDicomError

import pydicom
from torchvision.transforms.functional import to_tensor

import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from medclip.modeling_medclip import MedCLIPModel, PromptClassifier, MedCLIPVisionModel, MedCLIPVisionModelViT
from medclip.dataset import ImageTextContrastiveDataset, ZeroShotImageDataset

from medclip.dataset import SuperviseImageDataset, SuperviseImageCollator #new
from medclip.modeling_medclip import SuperviseClassifier


from medclip.dataset import ImageTextContrastiveCollator, ZeroShotImageCollator
from medclip.losses import ImageTextContrastiveLoss

from medclip.losses import ImageSuperviseLoss

from medclip.trainer import Trainer
from medclip.evaluator import Evaluator
from medclip import constants
from medclip.prompts import generate_class_prompts, generate_chexpert_class_prompts, generate_covid_class_prompts


# Set random seed for reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
os.environ['PYTHONASHSEED'] = str(seed)
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Set CUDA devices
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = "cpu"

# Training configurations
train_config = {
    'batch_size': 64, #64
    'num_epochs': 15,
    'warmup': 0.1,
    'lr': 1.0e-4, #2e-5,
    'weight_decay': 1e-6,
    'eval_batch_size': 64,
    'eval_steps': 500, #500 
    'save_steps': 500,
}

# Define class names for RSNA
class_names = ['label'] #0 and 1

# Define transformations for image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to 224x224
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1)), 
    transforms.Normalize(mean=[0.485, 0.485, 0.485], std=[0.229, 0.229, 0.229])  
])

# Custom dataset class for RSNA dataset they provided us
class RSNASuperviseImageDataset:
    def __init__(self, datalist, class_names, imgtransform=None):
        """
        Args:
            datalist (list): List of dataset identifiers (e.g., 'rsna-balanced-train').
            class_names (list): List of class names for classification.
            imgtransform (callable, optional): Transformations to apply to images.
        """
        self.transform = imgtransform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485], std=[0.229]),
        ])
        self.class_names = class_names

        df_list = []
        for data in datalist:
            filename = f'./local_data/{data}-meta.csv'
            print('Loading data from:', filename)
            df = pd.read_csv(filename)
            # Map column names from our csv to match MedCLIP's expectations
            df = df.rename(columns={'path': 'imgpath', 'label': class_names[0]})
            df_list.append(df)

        self.df = pd.concat(df_list, axis=0).reset_index(drop=True)
        #self.df = shuffle(self.df).reset_index(drop=True) #shuffle before sampling a subset for the smaller dataset in the next line
        #self.df = self.df[:800] #smaller set for easier debugging -> ~10%
        print("Label distribution:")
        print(self.df['label'].value_counts())

    def __len__(self):
        return len(self.df)


    def __getitem__(self, index):
        row = self.df.iloc[index]

        # Load and preprocess DICOM image
        #dicom = pydicom.dcmread(row['imgpath'])
        try:
            dicom = pydicom.dcmread(row['imgpath'])
        except InvalidDicomError:
            print(f"Skipping invalid DICOM file: {row['imgpath']}")
            return None
        img = dicom.pixel_array  # Extract pixel data
        img = Image.fromarray(img).convert("L")  # Convert to PIL Image (grayscale)
        img = self.transform(img)  # Apply transformations
        #print(f"Image shape after transformation: {img.shape}")

        img = img.unsqueeze(0) # Add channel dimension to make it [1, height, width]
        

        label = torch.tensor(row[self.class_names[0]], dtype=torch.float32)  # Use scalar label
    
        #label = pd.DataFrame([[row[col] for col in self.class_names]], columns=self.class_names)
        #label = torch.tensor([row[self.class_names[0]]], dtype=torch.float32)

        return img, label


# Initialize datasets and dataloaders
train_dataset = RSNASuperviseImageDataset(['rsna-balanced-train'], class_names=class_names, imgtransform=transform)
val_dataset = RSNASuperviseImageDataset(['rsna-balanced-val'], class_names=class_names, imgtransform=transform)

trainloader = DataLoader(
    train_dataset,
    batch_size=train_config['batch_size'],
    collate_fn=SuperviseImageCollator(mode="binary"),
    shuffle=True,
    num_workers=0,  # Adjust system's CPU capabilities #or more!! 
)

eval_dataloader = DataLoader(
    val_dataset,
    batch_size=train_config['eval_batch_size'],
    collate_fn=SuperviseImageCollator(mode="binary"),
    shuffle=False,
    num_workers=0, #or more!! 
)


# Build MedCLIP model and classifier
device = torch.device("cpu")
#model = MedCLIPModel(device=torch.device("cpu"))
model = MedCLIPModel(vision_cls=MedCLIPVisionModelViT, device=torch.device("cpu")) #use ViT #use CPU
model.to(device)

medclip_clf = SuperviseClassifier(
    vision_model=model,  
    num_class=1,         # Binary classification
    input_dim=512,       # output dim of vision model
    mode="binary",
    device=torch.device("cpu"), # CPU
)

print("Initial classifier bias:", medclip_clf.fc.bias)

medclip_clf.to(device)
#print("Using classifier:", medclip_clf)


# Build evaluator
evaluator = Evaluator(
    medclip_clf=medclip_clf,
    eval_dataloader=eval_dataloader,
    mode='binary'
)

image_supervise_loss = ImageSuperviseLoss(model=medclip_clf)


train_objectives = [
    (trainloader, image_supervise_loss, 1),
]



# Define save path for the model
model_save_path = './checkpoints/rsna_binary_classification'

"""
if __name__ == "__main__":
    # Train the model
    trainer = Trainer()
    trainer.train(
        model,
        train_objectives=train_objectives,
        warmup_ratio=train_config['warmup'],
        epochs=train_config['num_epochs'],
        optimizer_params={'lr': train_config['lr']},
        output_path=model_save_path,
        evaluation_steps=train_config['eval_steps'],
        weight_decay=train_config['weight_decay'],
        save_steps=train_config['save_steps'],
        evaluator=evaluator,
        eval_dataloader=eval_dataloader,
        use_amp=False,  # Disable AMP because I use CPU -> for GPU you can enable
    )

print('Training complete.')

"""


#########-----------------TESTING-----------------#########
test_dataset = RSNASuperviseImageDataset(['rsna-balanced-test'], class_names=class_names, imgtransform=transform)

test_dataloader = DataLoader(
    test_dataset,
    batch_size=train_config['eval_batch_size'],  # Use evaluation batch size
    collate_fn=SuperviseImageCollator(mode="binary"),
    shuffle=False,
    num_workers=0,  # Adjust based on system's CPU capabilities
)

# Load the final trained model
final_model_path = os.path.join(model_save_path, 'final_model.bin')
print(f"Loading the final model from {final_model_path}")
model.load_state_dict(torch.load(final_model_path, map_location=torch.device("cpu")))

# Set the model to evaluation mode
model.eval()

# Initialize evaluator for the test set
test_evaluator = Evaluator(
    medclip_clf=medclip_clf,
    eval_dataloader=test_dataloader,
    mode='binary'  # Ensure mode matches your task
)

# Evaluate on the test set
print("\n######### Testing on Test Set #########")
test_scores = test_evaluator.evaluate()

# Print test results
print("\n######### Final Test Results #########")
print(f"Confusion Matrix:")
print(f"TP: {test_scores['tp']}, TN: {test_scores['tn']}, FP: {test_scores['fp']}, FN: {test_scores['fn']}")
print(f"Accuracy: {test_scores['accuracy']:.4f}")
print(f"Recall: {test_scores['recall']:.4f}")
if 'val_loss' in test_scores:
    print(f"Test Loss: {test_scores['val_loss']:.4f}")



"""
#Original version from GitHUB repo
# set random seed
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
os.environ['PYTHONASHSEED'] = str(seed)
os.environ['TOKENIZERS_PARALLELISM']='false'

# set cuda devices
os.environ['CUDA_VISIBLE_DEVICES']='0'
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# set training configurations
train_config = {
    'batch_size': 100,
    'num_epochs': 10,
    'warmup': 0.1, # the first 10% of training steps are used for warm-up
    'lr': 2e-5,
    'weight_decay': 1e-4,
    'eval_batch_size': 256,
    'eval_steps': 1000,
    'save_steps': 1000,
}

# only pretrain on chexpert train data and mimic-cxr data
# do zero-shot training on chexpert-5x200 and iuxray
datalist = [
    'chexpert-train',
    'mimic-cxr-train',
]

transform = transforms.Compose([
                transforms.RandomHorizontalFlip(0.5),
                transforms.ColorJitter(0.2,0.2),
                transforms.RandomAffine(degrees=10, scale=(0.8,1.1), translate=(0.0625,0.0625)),
                transforms.Resize((256, 256)),
                transforms.RandomCrop((constants.IMG_SIZE, constants.IMG_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[constants.IMG_MEAN],std=[constants.IMG_STD])],
            )

traindata = ImageTextContrastiveDataset(datalist=datalist, imgtransform=transform)

train_collate_fn = ImageTextContrastiveCollator()

trainloader = DataLoader(traindata,
    batch_size=train_config['batch_size'],
    collate_fn=train_collate_fn,
    shuffle=True,
    pin_memory=True,
    num_workers=12,
    )

# build medclip model
model = MedCLIPModel(vision_cls=MedCLIPVisionModelViT)
model.cuda()

# build evaluator
cls_prompts = generate_chexpert_class_prompts(n=10) # ? 

val_data = ZeroShotImageDataset(['chexpert-5x200-val'],
    class_names=constants.CHEXPERT_COMPETITION_TASKS)

val_collate_fn = ZeroShotImageCollator(cls_prompts=cls_prompts,
    mode='multiclass')

eval_dataloader = DataLoader(val_data,
    batch_size=train_config['eval_batch_size'],
    collate_fn=val_collate_fn,
    shuffle=False,
    pin_memory=True,
    num_workers=4,
    )

medclip_clf = PromptClassifier(model) #sandra

#medclip_clf = SuperviseClassifier(model, num_class = 2, mode = "binary")

evaluator = Evaluator(
    medclip_clf=medclip_clf,
    eval_dataloader=eval_dataloader,
    mode='multiclass', 
)

# build loss models and start training
loss_model = ImageTextContrastiveLoss(model)
loss_model.cuda()
train_objectives = [
    (trainloader, loss_model, 1),
]
model_save_path = f'./checkpoints/vision_text_pretrain'
trainer = Trainer()
trainer.train(
    model,
    train_objectives=train_objectives,
    warmup_ratio=train_config['warmup'],
    epochs=train_config['num_epochs'],
    optimizer_params={'lr':train_config['lr']},
    output_path=model_save_path,
    evaluation_steps=train_config['eval_steps'],
    weight_decay=train_config['weight_decay'],
    save_steps=train_config['save_steps'],
    evaluator=evaluator,
    eval_dataloader=eval_dataloader,
    use_amp=True,
    )
print('done')
"""
