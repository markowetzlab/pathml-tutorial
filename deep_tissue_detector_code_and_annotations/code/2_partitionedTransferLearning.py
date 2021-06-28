import argparse
import copy
import datetime
import glob
import os
import pickle
import random
import time

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score

import numpy as np
from ImbalancedDatasetSampler import ImbalancedDatasetSampler
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Run training on tiles.')
parser.add_argument("-model", required=True, help="model file")
parser.add_argument("-tilefolder", required=True, help="tile folder")
parser.add_argument("-output", required=True, help="output folder")
args = parser.parse_args()
print(args)

now = datetime.datetime.now()
which_cnn = args.model

if which_cnn == 'vgg16' or which_cnn == 'resnet18' or which_cnn == 'squeezenet' or which_cnn == 'densenet' or which_cnn == "alexnet":
    patch_size = 224
if which_cnn == 'inceptionv3':
    patch_size = 299

data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(patch_size),
        transforms.RandomVerticalFlip(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(180),
        transforms.ColorJitter(
            brightness=.1, contrast=.1, saturation=.1, hue=.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(patch_size),
        # transforms.RandomVerticalFlip(),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# data_dir = '/media/gehrun01/work-io/cruk-phd-data/cytosponge/tiles-' + whichStain
data_dir = args.tilefolder

caseList = glob.glob("../wsis/*.svs") + glob.glob("../wsis/*.tif")
caseList = [os.path.split(case)[-1] for case in caseList]

random.seed(2355)  # 2355 for he, 23563 for tff3


random.shuffle(caseList)
# trainingPartitionSizes = [1,5,10,20,30,40,50,60,70,80,90]
# trainingPartitionSizes = [10, 20, 30, 40, 50, 60, 70, 80, 85]  # was 85
trainingPartitionSizes = [52]  # was 85
class_names = ['artifact', 'background', 'tissue']
val_cases = caseList[0:9]  # 15 for HE, 13 for TFF3
print(val_cases)
class_count = [0] * len(class_names)
class_count_val = [0] * len(class_names)
# Crawl through numbers

for caseName in caseList[-1 - 46:-1]:
    caseName = caseName.replace('.svs','').replace('.tif','')
    for classIdx, className in enumerate(class_names):
        #print(data_dir + '/' + caseName + '/' + className)
        path, dirs, files = next(
            os.walk(data_dir + '/' + caseName + '/' + className))
        file_count = len(files)
        class_count[classIdx] += file_count
print('Class names: ', class_names)
print('Sample count for training dataset: ', class_count)

for caseName in val_cases:
    caseName = caseName.replace('.svs','').replace('.tif','')
    for classIdx, className in enumerate(class_names):
        path, dirs, files = next(
            os.walk(data_dir + '/' + caseName + '/' + className))
        file_count = len(files)
        class_count_val[classIdx] += file_count
print('Sample count for validation dataset: ', class_count_val)


def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            epoch_ground_truth = []
            epoch_predictions = []
            running_loss = 0.0
            running_corrects = 0
            running_tp = {className: 0 for classIdx,
                          className in enumerate(class_names)}
            running_fp = {className: 0 for classIdx,
                          className in enumerate(class_names)}
            running_tn = {className: 0 for classIdx,
                          className in enumerate(class_names)}
            running_fn = {className: 0 for classIdx,
                          className in enumerate(class_names)}

            # Iterate over data.
            for inputs, labels in tqdm(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    if which_cnn == "inceptionv3" and phase == 'train':
                        outputs, aux = model(inputs)
                    else:
                        outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)

                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                #print('Prediction: ', preds)
                #print('Ground truth: ', labels.data)
                epoch_ground_truth = epoch_ground_truth + labels.data.tolist()
                epoch_predictions = epoch_predictions + preds.tolist()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                running_tp = {className: running_tp[className] + torch.sum((preds == classIdx) & (labels.data == classIdx)) for classIdx,
                              className in enumerate(class_names)}
                running_fp = {className: running_fp[className] + torch.sum(((preds == classIdx) ^ (labels.data == classIdx)) == 1) for classIdx,
                              className in enumerate(class_names)}
                running_tn = {className: running_tn[className] + torch.sum(((preds == classIdx) + (labels.data == classIdx)) == 0) for classIdx,
                              className in enumerate(class_names)}
                running_fn = {className: running_fn[className] + torch.sum(((preds == classIdx) ^ (labels.data == classIdx)) == -1) for classIdx,
                              className in enumerate(class_names)}

                #print(running_tp,running_fp,running_tn,running_fn)

                #print(running_tp, running_fp, running_tn, running_fn)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            epoch_acc = accuracy_score(epoch_ground_truth, epoch_predictions)
            epoch_weighted_acc = balanced_accuracy_score(epoch_ground_truth, epoch_predictions)  # accuracy accounting for class imbalance
            epoch_weighted_rec = recall_score(epoch_ground_truth, epoch_predictions, average='weighted')  # average recall accounting for class imbalance
            epoch_weighted_prec = precision_score(epoch_ground_truth, epoch_predictions, average='weighted')  # average precision accounting for class imbalance
            epoch_weighted_f1 = f1_score(epoch_ground_truth, epoch_predictions, average='weighted')  # average F1 score accounting for class imbalance

            epoch_rec = {className: running_tp[className].item() / (running_tp[className].item() + running_fn[className].item()) for classIdx,
                         className in enumerate(class_names)}
            epoch_prec = {className: running_tp[className].item() / (running_tp[className].item() + running_fp[className].item()) for classIdx,
                          className in enumerate(class_names)}
            # epoch_f1 = {className: (5 * 2) for classIdx,
            #            className in enumerate(class_names)}

            learningStatsCollection[phase].append(
                {'loss': epoch_loss, 'accuracy': epoch_acc, 'precision': epoch_prec, 'recall': epoch_rec,
                    'weighted_accuracy': epoch_weighted_acc,
                    'weighted_precision': epoch_weighted_prec,
                    'weighted_recall': epoch_weighted_rec,
                    'weighted_f1': epoch_weighted_f1})

            print('Phase Loss: {:.4f} Acc: {:.4f} Weighted Acc: {:.4f} Weighted Pre: {:.4f} Weighted Rec: {:.4f} Weighted F1: {:.4f} Rec: {:.4f} Pre: {:.4f}')
            print(phase, epoch_loss, epoch_acc, epoch_weighted_acc, epoch_weighted_prec, epoch_weighted_rec, epoch_weighted_f1, epoch_rec, epoch_prec)

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

            torch.save(model.state_dict(), os.path.join(args.output, 'partSize-' + str(len(train_cases)) + '-' + str(len(val_cases)
                                                                                                                              ) + '-' + str(which_cnn) + '_' + str(now.day) + '-' + str(now.month) + '-' + str(now.year) + '-epoch-' + str(epoch) + '-' + phase + '_ft.pt'))
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


for partitionSize in trainingPartitionSizes:
    learningStatsCollection = {'train': [], 'val': []}
    print("Starting training for partition size: " + str(partitionSize))
    train_cases = caseList[-1 - partitionSize:-1]

    train_dataset = torch.utils.data.ConcatDataset([datasets.ImageFolder(os.path.join(
        data_dir, trainCase.replace('.svs','').replace('.tif','')), data_transforms['train']) for trainCase in train_cases])
    val_dataset = torch.utils.data.ConcatDataset([datasets.ImageFolder(os.path.join(
        data_dir, valCase.replace('.svs','').replace('.tif','')), data_transforms['val']) for valCase in val_cases])

    # print(train_dataset.__len__())

    image_datasets = {'train': train_dataset, 'val': val_dataset}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    # print(dataset_sizes)
    # quit()
    dataloaders = {}

    if which_cnn == "resnet18":
        batch_size = 128
    elif which_cnn == "vgg16":
        batch_size = 48
    elif which_cnn == "inceptionv3":
        batch_size = 48
    elif which_cnn == "alexnet":
        batch_size = 64
    elif which_cnn == "squeezenet":
        batch_size = 256
    elif which_cnn == "densenet":
        batch_size = 84

    #dataloaders['train'] = torch.utils.data.DataLoader(image_datasets['train'], batch_size=batch_size, sampler=ImbalancedDatasetSampler(
    #    image_datasets['train']), num_workers=16)
    dataloaders['train'] = torch.utils.data.DataLoader(image_datasets['train'], batch_size=batch_size, shuffle=True, num_workers=16)
    dataloaders['val'] = torch.utils.data.DataLoader(
        image_datasets['val'], batch_size=batch_size, num_workers=16)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        print("Training on GPU")

    if which_cnn == "resnet18":
        model_ft = models.resnet18(pretrained=True)
        model_ft.fc = nn.Linear(512, len(class_names))
    elif which_cnn == "inceptionv3":
        model_ft = models.inception_v3(pretrained=True)
        num_ftrs = model_ft.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(768, len(class_names))
        model_ft.fc = nn.Linear(num_ftrs, len(class_names))
    elif which_cnn == "vgg16":
        model_ft = models.vgg16(pretrained=True)
        model_ft.classifier[6] = nn.Linear(4096, len(class_names))
    elif which_cnn == "densenet":
        model_ft = models.densenet121(pretrained=True)
        model_ft.classifier = nn.Linear(1024, len(class_names))
    elif which_cnn == "alexnet":
        model_ft = models.alexnet(pretrained=True)
        model_ft.classifier[6] = nn.Linear(4096, len(class_names))
    elif which_cnn == "squeezenet":
        model_ft = models.squeezenet1_1(pretrained=True)
        model_ft.classifier[1] = nn.Conv2d(
            512, len(class_names), kernel_size=(1, 1), stride=(1, 1))
        model_ft.num_classes = len(class_names)

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model_ft = nn.DataParallel(model_ft)
    model_ft = model_ft.to(device)

    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(
        optimizer_ft, step_size=7, gamma=0.1)

    model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                           num_epochs=25)

    pickle.dump(learningStatsCollection, open(os.path.join(args.output, 'learningCurve-' + str(which_cnn) + '-' +
                                                           str(len(train_cases)) + '-' + str(len(val_cases)) + '_' + str(now.day) + '-' + str(now.month) + '-' + str(now.year) + '.p'), 'wb'))
    torch.save(model_ft, os.path.join(args.output, 'partSize-' + str(len(train_cases)) + '-' + str(len(val_cases)
                                                                                                                      ) + '-' + str(which_cnn) + '_' + str(now.day) + '-' + str(now.month) + '-' + str(now.year) + '_ft.pt'))
    torch.save(model_ft.module.state_dict(), os.path.join(args.output, 'partSize-' + str(len(train_cases)) + '-' + str(len(val_cases)
                                                                                                                      ) + '-' + str(which_cnn) + '_' + str(now.day) + '-' + str(now.month) + '-' + str(now.year) + '_ft_state_dict.pt'))
