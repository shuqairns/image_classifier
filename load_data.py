import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict

def load_data(data_dir_path):
    train_dir = data_dir_path + '/train'
    valid_dir = data_dir_path + '/valid'
    test_dir = data_dir_path + '/test'

    data_transforms = transform_sets()
    # TODO: Load the datasets with ImageFolder
    image_datasets = {"train": datasets.ImageFolder(train_dir, transform=data_transforms["train"]),
                      "valid": datasets.ImageFolder(valid_dir, transform=data_transforms["valid_test"]),
                      "test": datasets.ImageFolder(test_dir, transform=data_transforms["valid_test"])}

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    dataloaders = {"train": torch.utils.data.DataLoader(image_datasets["train"], batch_size=64, shuffle=True),
                   "valid": torch.utils.data.DataLoader(image_datasets["valid"], batch_size=64, shuffle=True),
                   "test": torch.utils.data.DataLoader(image_datasets["test"], batch_size=64, shuffle=True)}
    
    return dataloaders, image_datasets
    
def transform_sets():
    # TODO: Define your transforms for the training, validation, and testing sets
    data_transforms = {"train": transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.RandomVerticalFlip(),
                                           transforms.Pad(4),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])]),
                       "valid_test": transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])}
    return data_transforms
