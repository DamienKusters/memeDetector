from __future__ import print_function, division
from PIL import Image
from modeltrainer import train_model

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt



alexnet = models.alexnet(pretrained=True)
num_c = 4096 # should be 4096 (based on alexnet source code)

alexnet.classifier[-1] = nn.Linear(num_c, 2) #Replace last (index: -1) final fully connected layer with our own, in this case with one output 

#alexnet = alexnet.to(device) #Move to GPU/CPU if needed
alexnet.cuda()

criterion = nn.MSELoss()

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(alexnet.parameters(), lr=0.001, momentum=0.9) #Optimizer crap I practically understand but not atm.

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1) #set training rate, just dont train TOO much

alexnet = train_model(alexnet, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=25)






img = Image.open("./yel.jpg")
img_t = transform(img)
batch_t = torch.unsqueeze(img_t, 0)


alexnet.eval()#put the model in evaluation mode
out = alexnet(batch_t)
#print(out.shape)


with open('./imagenet_classes.txt') as f:
    labels = [line.strip() for line in f.readlines()]


_, index = torch.max(out, 1)
percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100
print(labels[index[0]], percentage[index[0]].item())





