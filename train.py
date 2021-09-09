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


# get arguments from user
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default="ImageClassifier/flowers/train", help="Directory from which to pull data")
parser.add_argument('--save_dir', type=str, default="ImageClassifier/checkpoint.pth", help="Directory to save checkpoints")
parser.add_argument('--arch', type=str, default="vgg16", help="Architecture options: vgg16, vgg11")
parser.add_argument('--learning_rate', type=float, default=0.001, help="Learing rate of model")
parser.add_argument('--hidden_units', type=int, default=1000, help="Number of hiden units in model")
parser.add_argument('--epochs', type=int, default=5, help="Number of cycles used to train model")
parser.add_argument('--gpu', type=str, default='gpu', help="Options, gpu, cpu")

# add parser arguments to variables
inputs = parser.parse_args()

data_dir = inputs.data_dir
save_dir = inputs.save_dir
arch = inputs.arch
lr = inputs.learning_rate
hidden_layer = inputs.hidden_units
gpu = inputs.gpu
epochs = inputs.epochs

# get data and transforms it 
data_dir = 'ImageClassifier/flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                      transforms.RandomResizedCrop(224),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

test_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])


train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
test_data = datasets.ImageFolder(test_dir, transform=test_transforms)
valid_data = datasets.ImageFolder(valid_dir, transform=test_transforms)


trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
testloader = torch.utils.data.DataLoader(test_data, batch_size=64)
validloader = torch.utils.data.DataLoader(valid_data, batch_size=64)

# get model 

def get_classifier(arch, hidden_layer):
    
    """Use pretrained model and change the classifier for training.
    
    Arguments:
        arch (str): the name of a model
        hidden layer (int): Number of hiden units in model
    """
    
    if arch == "vgg11":
        model = models.vgg11(pretrained=True)
        start_size = 25088
    if arch == "vgg16":
        model = models.vgg16(pretrained=True)
        start_size = 25088
    else: 
        print("Only vgg16 and vgg11 can be used!")

    for param in model.parameters():
        param.requires_grad = False

    classifier = nn.Sequential(nn.Linear(start_size, hidden_layer),
                               nn.ReLU(),
                               nn.Dropout(0.2),
                               nn.Linear(hidden_layer, 500),
                               nn.ReLU(),
                               nn.Dropout(0.2),
                               nn.Linear(500, 250),
                               nn.ReLU(),
                               nn.Dropout(0.2),
                               nn.Linear(250, 102),
                               nn.LogSoftmax(dim=1))

    model.classifier = classifier
    
    return model


def train(model, gpu, epochs, lr):
    
    """Train model
    
    Arguments:
        model : the loaded model using the get_classifier function
        gpu (str): String indicating whether gpu or cpu must be used
        epochs (int): the  number of training cycles
        lr (float): the learning rate used
    """
    
    if gpu == 'gpu' and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    criterion = nn.NLLLoss()

    optimizer = optim.Adam(model.classifier.parameters(), lr)

    model.to(device)
    
    running_loss = 0
    steps = 0

    for epoch in range(epochs):
        for images, labels in trainloader:
            steps = steps + 1

            images, labels = images.to(device), labels.to(device)

            logprobs = model.forward(images)
            loss = criterion(logprobs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss = running_loss + loss.item()

            if steps % 40 == 0:
                test_loss = 0
                accuracy = 0
                model.eval()

                with torch.no_grad():
                    for images, labels in validloader:
                        images, labels = images.to(device), labels.to(device)
                        logprobs = model.forward(images)
                        batch_loss = criterion(logprobs, labels)
                        test_loss = test_loss + batch_loss.item()

                        probs = torch.exp(logprobs)
                        top_p, top_class = probs.topk(1, dim=1)
                        eqauls = top_class == labels.view(*top_class.shape)
                        accuracy = accuracy + torch.mean(eqauls.type(torch.FloatTensor)).item()

                        print(f"Epoch {epoch+1}/{epochs}.. "
                              f"Train loss: {running_loss/40:.3f}.. "
                              f"Validation loss: {test_loss/len(testloader):.3f}.. "
                              f"Validation accuracy: {(accuracy/len(testloader))*100:.3f}")

                        running_loss = 0

                        model.train()
                        
                        
# train model
choosen_model = get_classifier(arch, hidden_layer)
train(choosen_model, gpu, epochs, lr)

# save checkpoints
choosen_model.class_to_idx = train_data.class_to_idx
checkpoint = {'arch': arch,
              'lr': lr,
              'hidden_layer': hidden_layer,
              'gpu': gpu,
              'epochs': epochs,
              'classifier': choosen_model.classifier,
              'state_dict': choosen_model.state_dict(),
              'class_to_idx': choosen_model.class_to_idx}

torch.save(checkpoint, save_dir)
