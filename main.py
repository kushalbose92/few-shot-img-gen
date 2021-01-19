import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import torchvision
from torch.autograd import Variable
import random
from sklearn.manifold import TSNE
from sklearn import preprocessing
import datetime
import os
import copy
import torchvision.transforms as transforms
import learn2learn as l2l

from dataloader import MNISTLoader, OmniglotLoader, CIFAR10Loader, Fashion_MNISTLoader
from train import ModelTrainer
from eval import ModelEvaluation
from utils import UtilFunctions


num_shot = 5
num_query = 5
num_ways = 3
alpha = 500.0
beta = 300.0
num_support = num_shot * num_ways
num_test = num_query * num_ways
num_samples = (num_shot + num_query) * num_ways
latent_dims = 2
num_train_tasks = 50000
num_test_tasks = 500
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


data_loader = OmniglotLoader(num_ways, num_shot, num_query, num_train_tasks, num_test_tasks)

train_tasks = data_loader.train_task_set()
test_tasks = data_loader.test_task_set()


# alpha - Intra-Clustering Weighting Constant
# beta -  Inter-Clustering Weighting Constant
# If ways = 1 then put alpha = 0.0

trainer = ModelTrainer(
                    train_tasks = train_tasks,
                    shots = num_shot,
                    ways = num_ways,
                    maml_lr = 0.001,
                    base_lr = 0.01,
                    alpha = alpha,
                    beta = beta,
                    meta_train_batch = 32,
                    meta_train_epochs = 1,
                    steps_per_task = 1,
                    latent_dims = latent_dims,
                    device = device)

MODEL_PATH = trainer.train()
# MODEL_PATH = os.getcwd() + "/saved_models/omniglot_3-way-5-shot.pt"

evaluation = ModelEvaluation(
                            test_tasks = test_tasks,
                            shots = num_shot,
                            ways = num_ways,
                            base_lr = 0.01,
                            alpha = alpha,
                            beta = beta,
                            meta_test_batch = 1,
                            meta_test_epochs = 1,
                            steps_per_task = 50,
                            latent_dims = latent_dims,
                            device = device)

evaluation.eval(MODEL_PATH)

