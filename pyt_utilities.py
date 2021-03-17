# pyt_utilities.py (python3)
# utilities for CNN training with pytorch;  data preparation, training, evaluation
# Catch+ Release / Return to Bali
# FEB 2020
# sources:
# https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html#load-data
# https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html

# this module requires a GPU computer

#updated normalization mean and std:
#mean [0.4597244970012637, 0.4887084808460421, 0.46925360649661096]
#std [0.20728858675971737, 0.2048932794469992, 0.21645177513430724]
#-------------------------------------------------------------------------------
import os, sys, time, random
import torch, torchvision
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torchvision import models, datasets, transforms
import torch.nn.functional as F
from torch import utils
import torch.optim as optim
from PIL import Image
import numpy
import array
import urllib, glob, shutil
from shutil import copyfile, copy
from copy import deepcopy
import matplotlib.pyplot as plt
#-------------------------------------------------------------------------------
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.4597, 0.4887, 0.4692], [0.2072, 0.2048, 0.2164]) #bali26
        #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.4597, 0.4887, 0.4692], [0.2072, 0.2048, 0.2164]) #bali26
        #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

predict_transform = transforms.Compose([
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.4597, 0.4887, 0.4692], [0.2072, 0.2048, 0.2164]) #bali26
    #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

#-------------------------------------------------------------------------------
#simple three layer CNN for 224 x 224 input; reduced alexnet; untrained by default
#https://github.com/pytorch/vision/blob/master/torchvision/models/alexnet.py
class vanillanet(torch.nn.Module):
    def __init__(self, num_classes):
        super(vanillanet, self).__init__()

        self.cnn_layers = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=3, stride=2),

            torch.nn.Conv2d(64, 192, kernel_size=5, padding=2),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=3, stride=2),

            torch.nn.Conv2d(192, 256, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=3, stride=2),
        )

        self.avgpool = torch.nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = torch.nn.Sequential(
            torch.nn.Dropout(),
            torch.nn.Linear(256 * 6 * 6, 4096),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.cnn_layers(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return (x)

#-------------------------------------------------------------------------------
def prune_imageset(datapath, categories, limit, offset, randomprune):
    for i in range(0, len(categories)):
        files = list(filter(os.path.isfile, glob.glob(datapath + categories[i] + '/' + "*")))

        if(randomprune == True):
            random.shuffle(files)
        else:
            files.sort(key=lambda x: os.path.getmtime(x))

        for i in range (0, len(files)):
            if(i < limit):
                pass
            else:
                print("random?, getting rid of: ", randomprune, files[i])
                os.remove(files[i])

#-------------------------------------------------------------------------------
def create_train_val_sets(datapath, categories, percentage):
    train = datapath + 'train/'
    val = datapath + 'val/'

    if not os.path.exists(train):
        os.mkdir(train)
        for k in categories:
            os.mkdir(train + k)

    if not os.path.exists(val):
        os.mkdir(val)
        for k in categories:
            os.mkdir(val + k)

    os.chdir(datapath)
    for i in range(0, len(categories)):
        files = list(filter(os.path.isfile, glob.glob(datapath + categories[i] + '/' + "*")))
        files.sort(key=lambda x: os.path.getmtime(x))

        traininglimit = int(percentage*len(files))
        print('\ncategory: ', categories[i])
        print('number files for training: ', traininglimit)
        print('number files for validation: ', (len(files) - traininglimit))

        for j in range (0, len(files)):
            filename = files[j].split('/')[-1]
            filecatname = categories[i] + '/' + filename
            if(j < traininglimit):
                filecatnametrain = train + filecatname
                shutil.copy(files[j], filecatnametrain)
            else:
                filecatnameval = val + filecatname
                shutil.copy(files[j], filecatnameval)

#-------------------------------------------------------------------------------
def train_model(checkpointname, model, dataloaders, dataset_sizes, criterion, optimizer, scheduler, device, num_epochs, output):
    since = time.time()
    #best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    e_val_loss = []; e_train_loss = []
    e_val_acc = []; e_train_acc = []

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                print('training...')
                model.train()  # Set model to training mode
            else:
                print('evaluating...')
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0; running_corrects = 0
            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward; track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            #loss = '{:.4f}'.format(epoch_loss)
            #acc = '{:.4f}'.format(epoch_acc)
            loss = float('{:.4f}'.format(epoch_loss))
            acc = float('{:.4f}'.format(epoch_acc))

            if(phase == 'train'):
                e_train_loss.append(loss)
                e_train_acc.append(acc)

            if(phase == 'val'):
                e_val_loss.append(loss)
                e_val_acc.append(acc)

            if (phase == 'val' and epoch_acc > best_acc):
                best_acc = epoch_acc
                best_model_wts = deepcopy(model.state_dict())
                checkpoint = {'model': model,'state_dict': model.state_dict(), 'optimizer' : optimizer.state_dict()}

        print()

    time_elapsed = time.time() - since
    model.load_state_dict(best_model_wts)
    torch.save(checkpoint, checkpointname)

    plotresults(e_val_loss, e_train_loss, e_val_acc, e_train_acc, output)

    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    print('Saved best checkpoint')
    print('Plotted training results')
    print('Returning best model')
    return (model)

#-------------------------------------------------------------------------------
def predict_image(image_path, model, transform, class_names, tk):
    img = Image.open(image_path)
    img_t = transform(img)
    batch_t = torch.unsqueeze(img_t, 0).cuda()
    model.eval()
    output = model(batch_t)

    predictions = output.topk(tk,1,largest=True,sorted=True)
    _, index = torch.max(output, 1)
    t_percentage = torch.nn.functional.softmax(output, dim=1)[0] * 100
    percentage = t_percentage[index[0]].item()
    percentage = '%.2f'%(percentage)
    category = class_names[index[0]]

    return(predictions, percentage, category)

#-------------------------------------------------------------------------------
def check_topN(class_names, topNlist, tk, input):
    topN = 0
    for i in range (0, len(topNlist)):
        if(class_names[topNlist[i]] == input):
            topN = 1
            break
    return(topN)

#-------------------------------------------------------------------------------
def load_checkpoint(filepath):
    print('got this far into loading checkpoint...')
    checkpoint = torch.load(filepath)
    model = checkpoint['model']
    print('now got this far into loading checkpoint...')
    model.load_state_dict(checkpoint['state_dict'])
    for parameter in model.parameters():
        parameter.requires_grad = False

    model.eval()
    return (model)

#-------------------------------------------------------------------------------
def write2file(filename, comment):
    file = open(filename, "a")
    file.write(comment)
    file.write('\n')
    file.close()

#-------------------------------------------------------------------------------
def plotresults(e_val_loss, e_train_loss, e_val_acc, e_train_acc, output):

    ind = [i for i in range(len(e_val_loss))]
    best = 'best eval accuracy: ' + str(numpy.max(e_val_acc)) + '; best train accuracy: ' + str(numpy.max(e_train_acc))
    print(best)

    fig, axs = plt.subplots(4, figsize=(15, 15))
    axs[0].plot(ind, e_val_loss,  marker='x', markersize=8, c='r', linestyle = '--', linewidth=1)
    axs[1].plot(ind, e_val_acc,  marker='x', markersize=5, c='r', linestyle = '--', linewidth=1)
    axs[2].plot(ind, e_train_loss,  marker='s', markersize=6, c='r', linestyle = '--', linewidth=1)
    axs[3].plot(ind, e_train_acc,  marker='s', markersize=3, c='r', linestyle = '--', linewidth=1)

    for i in range (0,4):
        axs[i].yaxis.set_major_locator(plt.MaxNLocator(8))
        axs[i].set_ylabel('score')
        axs[i].grid()
        axs[i].set_xlabel('training epochs')

    for ax in axs.flat:
        ax.label_outer()

    text = 'Evaluation loss (sm) and accuracy (lg) [cross]; \n Training loss (sm) and accuraccy (lg) [square]; \n' + best + '\n'
    fig.suptitle(text, fontsize=18)
    fig.subplots_adjust(top=0.9)
    plt.savefig(output)

#-------------------------------------------------------------------------------
def autolabel(bars, ax, fs):
    for rect in bars:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,'%d' % int(height),ha='center', va='bottom', fontsize=fs)
#-------------------------------------------------------------------------------
