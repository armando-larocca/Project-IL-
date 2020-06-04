import torch
import random
import os
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from PIL import Image
import torch.optim as optim 
import torchvision.datasets as dsets
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Subset, DataLoader
import sys
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def lb(Subset):
  it =list(Subset)
  uscita = []
  for coppie in it:
    uscita.append(coppie[2])
  return uscita


def class_images(Subset, label):
  ite =list(Subset)
  us = []
  ind = []
  for coppia in ite:
    if coppia[2] == label:
      us.append(coppia[1])
      ind.append(coppia[0])
  return ind, us
 
 
 
  def add(Subset,exemplar_images, exemplar_labels, e):
   ite =list(Subset)
   coppie = []
   for i in range(0, len(exemplar_images)):
     coppia = []
     coppia.append(e)
     coppia.append(exemplar_images[i])
     coppia.append(exemplar_labels[i])
     ite.append(coppia)
     e = e+1
   return e, ite



def show_images(images):
    N = images.shape[0]
    fig = plt.figure(figsize=(1, N))
    gs = gridspec.GridSpec(1, N)
    gs.update(wspace=0.05, hspace=0.05)
    for i, img in enumerate(images):
        img = np.transpose(img, (1,2,0))
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(img)
    plt.show()
