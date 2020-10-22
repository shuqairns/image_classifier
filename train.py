# Some code from https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict
import argparse
import numpy as np

from PIL import Image

from workspace_utils import keep_awake, active_session
from init_model import initialize_model
from load_data import load_data


def train(data_directory, arch, num_classes, hidden_units, learning_rate, momentum, epochs, gpu, save_dir):
    device = torch.device("cuda" if gpu and torch.cuda.is_available() else "cpu")
    
    model = initialize_model(arch, num_classes, True, hidden_units, True).to(device)
    print(device)
    criterion = nn.NLLLoss()
    optimizer = init_optimier(model, learning_rate, momentum)
#     optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
#     model.to(device)
    
    dataloaders, image_datasets  = load_data(data_directory)
    
    steps = 0
    print_every = 5

    with active_session():
        for e in range(epochs):
            running_loss = 0
            for inputs, labels in dataloaders["train"]:
                steps +=1
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model.forward(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                if steps % print_every == 0:
                    test_loss = 0
                    accuracy = 0
                    model.eval()
                    with torch.no_grad():
                        for inputs, labels in dataloaders["valid"]:
                            inputs, labels = inputs.to(device), labels.to(device)
                            logps = model.forward(inputs)
                            batch_loss = criterion (logps, labels)
                            test_loss += batch_loss.item()

                            # Calculate accuracy 
                            ps = torch.exp(logps)
                            top_p, top_class = ps.topk(1, dim=1)
                            equals = top_class == labels.view(*top_class.shape)
                            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                    valLen = len(dataloaders["valid"])
                    print(f"Epoch {e+1}/{epochs}.. "
                          f"Train loss: {running_loss/print_every:.3f}.. "
                          f"validation loss: {test_loss/valLen:.3f}.. "
                          f"validation accuracy: {accuracy/valLen:.3f}")
                    running_loss = 0
                    model.train()
                    
    save_checkpoint(save_dir, model, optimizer, arch, image_datasets)

def init_optimier(model_ft, learning_rate, momentum):
    params_to_update = model_ft.parameters()
    print("Params to learn:")
    params_to_update = []
    for name,param in model_ft.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            print("\t",name)

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(params_to_update, lr=learning_rate, momentum=momentum)
    return optimizer_ft
    
def save_checkpoint(save_dir, model, optimizer, arch, image_datasets):
    model.class_to_idx = image_datasets['train'].class_to_idx
    checkpoint = {'model': model,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()}
    torch.save(checkpoint, save_dir + '/' + arch + '.checkpoint.pth')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trains flower classifier")
    parser.add_argument('data_directory')
    parser.add_argument('--save_dir', action='store', dest='save_dir', required=True)
    parser.add_argument('--arch', action='store', dest='arch', required=False, default="densenet", choices=['resnet', 'alexnet', 'vgg', 'squeezenet', 'densenet'])
    parser.add_argument('--num_classes', action='store', dest='num_classes', required=False, default=102)
    parser.add_argument('--learning_rate', action='store', dest='learning_rate', required=False, default=0.003)
    parser.add_argument('--momentum', action='store', dest='momentum', required=False, default=0.9)
    parser.add_argument('--hidden_units', action='store', dest='hidden_units', required=False, default=512)
    parser.add_argument('--epochs', action='store', dest='epochs', required=False, default=10)
    parser.add_argument('--gpu', action='store_true', dest='gpu', required=False)

    _args = parser.parse_args()
    print(_args)

    train(_args.data_directory, _args.arch, int(_args.num_classes), int(_args.hidden_units), float(_args.learning_rate), float(_args.momentum), int(_args.epochs), _args.gpu, _args.save_dir)
