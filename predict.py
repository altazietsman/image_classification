import argparse
import pandas as pd
import numpy as np
import torch
from torchvision import transforms, datasets, models
from collections import OrderedDict
from torch import nn
from torch import optim
import torch.nn.functional as F
import json
from PIL import Image

# get arguments
parser = argparse.ArgumentParser()

parser.add_argument('--top_k', type=int, default=3, help="The number of top predictions")
parser.add_argument('--category_names', type=str, default='ImageClassifier/cat_to_name.json', help="File with category names")
parser.add_argument('--gpu', type=str, default='gpu', help="Options, gpu, cpu")
parser.add_argument('--test_dir', type=str, default="ImageClassifier/flowers/test/1/image_06743.jpg", help="Directory from which to pull test data")
parser.add_argument('--checkpoint', type=str, default='ImageClassifier/checkpoint.pth', help="Checkpoint file used to rebuild trained model")

# add parser arguments to variables
inputs = parser.parse_args()
test_dir = inputs.test_dir
top_k = inputs.top_k
category_names = inputs.category_names
gpu = inputs.gpu
checkpoint = inputs.checkpoint

with open(category_names, 'r') as f:
    cat_to_name = json.load(f)

# rebuild model
def load_model(filepath):
    """Load and rebuild model
    
    Arguments:
        filepath (str): Directory where checkpoints of model is stored"""
    
    checkpoint = torch.load(filepath)
    
    arch = checkpoint['arch']
    classifier = checkpoint['classifier']
    state_dict = checkpoint['state_dict']
    class_to_idx = checkpoint['class_to_idx']

    if arch == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif arch == 'vgg11':
        model = models.vgg11(pretrained=True)

    model.classifier = classifier
    model.class_to_idx = class_to_idx
    model.load_state_dict(state_dict)

    for param in model.parameters():
        param.requires_grad = False

    return model


model = load_model(checkpoint)

# process image
def process_image(image):
    
    """Process image
    
    Arguments:
        image (str): directory path with image file to test
    """

    im = Image.open(image)    
    transform = transforms.Compose([transforms.Resize(256),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406],
                                                          [0.229, 0.224, 0.225])])    
    array = transform(im).float()    
    return array

def predict(image=test_dir, model=model, gpu=gpu, topk=top_k):
    
    """Make predictions
    
    Arguments:
        image (str): the directory where image to test is
        model: the trained model
        gpu (str): specify is gpu or cpu must be used
        topk (int): the top number of predictions based on probabilities    
    """
    if gpu == 'gpu' and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    image = process_image(image)
    image = image.float().unsqueeze_(0)
    
    with torch.no_grad():
        images = image.to(device)
        model = model.to(device)
        logprobs = model.forward(images)
        
    predictions = F.softmax(logprobs.data, dim=1)
    
    probs, ind = predictions.topk(topk)
    
    probs = probs.cpu().numpy()[0]
    indices = ind.cpu().numpy()[0]
    
    idx_to_class = {v:k for k, v in model.class_to_idx.items()}
    classes = [idx_to_class[x] for x in indices]
    names = [cat_to_name[x] for x in classes]
    
    print("Top classes:", classes)
    print("Top Categories:", names)
    print("Top probabilities:", probs)
    
    return probs, classes, names
    
# make predictions
predict(image=test_dir, model=model, gpu=gpu, topk=top_k)
