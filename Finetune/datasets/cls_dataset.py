import os
import numpy as np
import pandas as pd
import torch
from .utils import get_imgs, read_from_dicom
from torch.utils.data import Dataset
import ast

np.random.seed(42)

class RSNAImageClsDataset(Dataset):
    def __init__(self, config,split='train',transform=None,data_pct=1.0) -> None:
        super().__init__()
        if split == 'train':
            csv_path = config['dataset']['train_csv']
        elif split == 'valid':
            csv_path = config['dataset']['valid_csv']
        else:
            csv_path = config['dataset']['test_csv']
        self.df = pd.read_csv(csv_path)
        self.transform = transform
        if data_pct != 1.0:
            self.df = self.df.sample(frac=data_pct, random_state=42)
        self.imsize = config['cls']['img_size']
        print('Dataset size of split {}:'.format(split), len(self.df))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        # get image
        img_path = row["path"]
        x = read_from_dicom(img_path, self.imsize, self.transform)
        y = float(row["label"])
        y = torch.tensor([y])

        return x, y, img_path
    
class ChexPertImageClsDataset(Dataset):
    def __init__(self, config,split='train',transform=None,data_pct=1.0) -> None:
        super().__init__()
        if split == 'train':
            csv_path = config['dataset']['train_csv']
        elif split == 'valid':
            csv_path = config['dataset']['valid_csv']
        else:
            csv_path = config['dataset']['test_csv']
        self.df = pd.read_csv(csv_path)
        self.img_type = config['dataset']['img_type']

        CHEXPERT_VIEW_COL = config['dataset']['CHEXPERT_VIEW_COL']
        CHEXPERT_PATH_COL=  config['dataset']['CHEXPERT_PATH_COL']
        CHEXPERT_DATA_DIR = config['dataset']['dataset_dir']
        CHEXPERT_COMPETITION_TASKS = config['dataset']['CHEXPERT_COMPETITION_TASKS']
        CHEXPERT_UNCERTAIN_MAPPINGS = config['dataset']['CHEXPERT_UNCERTAIN_MAPPINGS']
        # import pdb; pdb.set_trace()
         # filter image type
        if self.img_type != "All":
            self.df = self.df[self.df[CHEXPERT_VIEW_COL] == self.img_type]

        self.transform = transform

        if data_pct != 1.0:
            self.df = self.df.sample(frac=data_pct, random_state=42)
        print('Dataset size of split {}:'.format(split), len(self.df))
        self.imsize = config['cls']['img_size']

        # get path
        self.df[CHEXPERT_PATH_COL] = self.df[CHEXPERT_PATH_COL].apply(
            lambda x: os.path.join(
                CHEXPERT_DATA_DIR, "/".join(x.split("/")[1:]))
        )

        # fill na with 0s
        self.df = self.df.fillna(0) # if na then fill with 0 (no observation)
        uncertain_mask = {k: -1 for k in CHEXPERT_COMPETITION_TASKS} # {'Atelectasis': -1, 'Cardiomegaly': -1, 'Consolidation': -1, 'Edema': -1, 'Pleural Effusion': -1}
        self.df = self.df.replace(uncertain_mask, CHEXPERT_UNCERTAIN_MAPPINGS)

        self.path = self.df["Path"].values
        self.labels = self.df.loc[:, CHEXPERT_COMPETITION_TASKS].values # (185779, 5) one sample array([0., 0., 0., 1., 1.])
       

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        # import pdb; pdb.set_trace()
        img_path = self.path[index]
        x = get_imgs(img_path, self.imsize, self.transform)
        y = self.labels[index]
        y = torch.tensor(y)
        return x, y


# import torch
# from torch.utils.data import DataLoader
# from torchvision import transforms
# import matplotlib.pyplot as plt

# # Assuming you have `RSNAImageDataset` class imported and defined

# # Define some basic transformations (resize, convert to tensor, normalize)
# transform = transforms.Compose([
#     transforms.RandomCrop(224),  
#     transforms.ToTensor(),  
#     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  
# ])

# # Instantiate the dataset
# csv_path = '/u/home/lj0/Code/VLP-Seminars/annotations/train.csv'  # Path to your CSV file
# dataset = RSNAImageDataset(csv_path=csv_path, transform=transform, data_pct=0.1, imsize=256)  # Using 10% of data

# # Create DataLoader for batching
# dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# # Iterate over a batch of data
# for batch_idx, (images, labels) in enumerate(dataloader):
#     import pdb; pdb.set_trace()
#     print(f"Batch {batch_idx + 1}:")
#     print(f"Images shape: {images.shape}")  # Should be [batch_size, channels, height, width]
#     print(f"Labels: {labels}")
#     break  # Remove this if you want to loop through all batches
