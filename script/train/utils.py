import os
import torch
import torchvision
from PIL import Image
from torch.utils.data import Dataset
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd


def rescale(x, old_range, new_range, clamp=False):
    old_min, old_max = old_range
    new_min, new_max = new_range
    x -= old_min
    x *= (new_max - new_min) / (old_max - old_min)
    x += new_min
    if clamp:
        x = x.clamp(new_min, new_max)
    return x


def plot_images(images):
    plt.figure(figsize=(32, 32))
    plt.imshow(torch.cat([torch.cat([i for i in images.cpu()], dim=-1)], dim=-2).permute(1, 2, 0).cpu())
    plt.show()


def save_images(images, path, **kwargs):
    images = rescale(images, (-1, 1), (0, 255), clamp=True)
    grid = torchvision.utils.make_grid(images, **kwargs)
    # (Batch_Size, Channel, Height, Width) -> (Batch_Size, Height, Width, Channel)
    nd_array = grid.permute(0, 2, 3, 1).to('cpu', torch.uint8).numpy()
    # nd_array = grid.permute(1, 2, 0).to('cpu').numpy()
    im = Image.fromarray(nd_array)
    im.save(path)

# Only for images
# def get_data(args):
#     transforms = torchvision.transforms.Compose([
#         torchvision.transforms.Resize(80),  # args.image_size + 1/4 *args.image_size
#         torchvision.transforms.RandomResizedCrop(args.image_size, scale=(0.8, 1.0)),
#         torchvision.transforms.ToTensor(),
#         torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
#     ])
#     dataset = torchvision.datasets.ImageFolder(args.dataset_path, transform=transforms)
#     dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
#     return dataloader


class ImageIngredientPromptDataset(Dataset):
    def __init__(self, dataframe, root_dir, transform=None):
        self.dataframe = dataframe
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.dataframe.iloc[idx, 0])
        image = Image.open(img_name).convert("RGB")
        if self.transform:
            image = self.transform(image)

        caption = self.dataframe.iloc[idx, 1]
        unique_index = self.dataframe.iloc[idx, 2]

        return image, caption, unique_index


def get_data(args):
    # Read the Excel file containing image file names and captions
    try:
        dataframe = pd.read_excel(args.excel_path, engine="openpyxl")
        dataframe = dataframe.head(args.dataset_set)
        dataframe = dataframe.reset_index().drop(['index'], axis=1)
        dataframe['unique_index'] = dataframe.index
        dataframe = dataframe[['image_id', 'ingredient_v2', 'unique_index']]
    except ModuleNotFoundError:
        dataframe = pd.read_excel(args.excel_path)

    # transforms = torchvision.transforms.Compose([
    #     torchvision.transforms.Resize(80),  # args.image_size + 1/4 *args.image_size
    #     torchvision.transforms.RandomResizedCrop(args.image_size, scale=(0.8, 1.0)),
    #     torchvision.transforms.ToTensor(),
    #     torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    # ])

    transforms = torchvision.transforms.Compose([
        torchvision.transforms.RandomResizedCrop(args.image_size, scale=(0.8, 1.0), interpolation=torchvision.transforms.InterpolationMode.BILINEAR),
        torchvision.transforms.ToTensor()
    ])

    dataset = ImageIngredientPromptDataset(dataframe, args.image_dataset_path, transform=transforms)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    return dataloader


class CustomDataset(Dataset):
    def __init__(self, latent_file, context_file):
        self.latent_file = latent_file
        self.context_file = context_file
        self.length = len(np.load(self.latent_file, mmap_mode='r'))  # Length of dataset

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        latent_data = np.load(self.latent_file, mmap_mode='r')
        context_data = np.load(self.context_file, mmap_mode='r')

        # Load specific samples only
        latent_tensor = torch.from_numpy(latent_data[idx].copy())  # Create a writable copy
        context_tensor = torch.from_numpy(context_data[idx].copy())  # Create a writable copy

        # Delete memory-mapped file objects to close them
        del latent_data
        del context_data

        return latent_tensor, context_tensor


def custom_collate(batch):
    latent_tensors, context_tensors = zip(*batch)
    return torch.stack(latent_tensors), torch.stack(context_tensors)


def setup_logging(run_name):
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs(os.path.join("models", run_name), exist_ok=True)
    os.makedirs(os.path.join("results", run_name), exist_ok=True)



