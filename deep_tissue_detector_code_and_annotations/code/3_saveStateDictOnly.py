import argparse
import copy
import datetime
import glob
import os
import pickle
import random
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision

parser = argparse.ArgumentParser(description='Convert model file to state dict.')
parser.add_argument("-model", required=True, help="model file")
parser.add_argument("-output", required=True, help="state dict file")
args = parser.parse_args()

now = datetime.datetime.now()

model = torch.load(args.model)
torch.save(model.state_dict(), args.output)
