import sys
import torch
from torchvision import transforms
from collections import OrderedDict
import numpy as np
from PIL import Image
import json
import argparse

cat_to_name = []
    
def predict(image_path, checkpointPath, topk, category_names_file, gpu):
    
    device = torch.device("cuda" if gpu and torch.cuda.is_available() else "cpu")
    print(device)
    
    model = load_checkpoint(checkpointPath).to(device).eval()
        
    img = process_image(image_path).unsqueeze(0).to(device)
    
    with torch.no_grad():
        logits = model.forward(img)
        
    ps = torch.exp(logits)
    top_p, top_class = ps.topk(topk, dim=1)
    class_reverse = {v: k for k, v, in model.class_to_idx.items()} 
    top_p, top_class = top_p.cpu().numpy(), top_class.cpu().numpy()
    
    cat_to_name = get_cat_names(category_names_file)
    categ = []
    for i in range(topk):
        categ.append(cat_to_name[class_reverse[top_class[0][i]]])
        
    return categ, top_p

def get_cat_names(path):
    with open(path, 'r') as f:
        cat_to_name = json.load(f)
    return cat_to_name

''' Loads the checkpoint to build the model'''
def load_checkpoint(filepath):
    if torch.cuda.is_available():
        map_location=lambda storage, loc: storage.cuda()
    else:
        map_location='cpu'

    checkpoint = torch.load(filepath, map_location=map_location)
    model = checkpoint['model']
    model.load_state_dict(checkpoint['model_state_dict'])
    return model

''' Scales, crops, and normalizes a PIL image for a PyTorch model,
    returns an Numpy array
'''
def process_image(image):    
    img_transforms = transforms.Compose([
            transforms.Resize(255),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], 
                         [0.229, 0.224, 0.225])
        ])
    
    image = img_transforms(Image.open(image))
    return image



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classifies flowers")
    parser.add_argument('input')
    parser.add_argument('checkpoint')
    parser.add_argument('--top_k', action='store', dest='top_k', required=False, default=5)
    parser.add_argument('--category_names', action='store', dest='category_names', required=True)
    parser.add_argument('--gpu', action='store_true', dest='gpu', required=False)
    
    _args = parser.parse_args()

    classes, probs = predict(_args.input, _args.checkpoint, int(_args.top_k), _args.category_names, _args.gpu)
    print(classes)
    print(probs)
