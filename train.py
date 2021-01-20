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

from model import ImageGenerator
from utils import UtilFunctions


# Class to train defined model
class ModelTrainer():
    def __init__(self, train_tasks, shots, ways, maml_lr, base_lr, alpha, beta, meta_train_batch, meta_train_epochs, steps_per_task, latent_dims, device):
        self.train_tasks = train_tasks
        self.shots = shots
        self.ways = ways
        self.maml_lr = maml_lr
        self.base_lr = base_lr
        self.alpha = alpha
        self.beta = beta
        self.meta_train_batch = meta_train_batch
        self.meta_train_epochs = meta_train_epochs
        self.steps_per_task = steps_per_task
        self.latent_dims = latent_dims
        self.device = device

    # training of model
    def train(self):

        # Definition of meta model 
        img_gen = ImageGenerator(self.latent_dims, self.ways, self.shots)
        img_gen.to(self.device)
        img_gen_meta = l2l.algorithms.MAML(img_gen, lr = self.base_lr, first_order = True)
        opti_meta = optim.Adam(img_gen_meta.parameters(), lr = self.maml_lr)

        print("Status: Training Stage \n")

        # Outer loop 
        for i in range(self.meta_train_epochs):

            meta_loss = 0.0
            opti_meta.zero_grad()
            for t_idx in range(self.meta_train_batch):

                img_gen_learner = img_gen_meta.clone()
                train_task = self.train_tasks.sample()

                data, labels = train_task
                data = data.to(self.device)
                labels = labels.to(self.device)

                adaptation_indices = np.zeros(data.size(0), dtype=bool)
                adaptation_indices[np.arange(self.shots * self.ways)*2] = True
                evaluation_indices = torch.from_numpy(~adaptation_indices).to(self.device)
                adaptation_indices = torch.from_numpy(adaptation_indices).to(self.device)

                support_set, support_labels = data[adaptation_indices], labels[adaptation_indices]
                query_set, query_labels = data[evaluation_indices], labels[evaluation_indices]

                # Inner loop
                for step in range(self.steps_per_task):

                    # Support set as input to model 
                    support_recon, latent_proto, latent_radius, latent_feat = img_gen_learner(support_set, gamma = 1.0)

                    # Loss definitions
                    recon_loss = UtilFunctions().loss_reconstruction(support_set, support_recon)
                    radius_loss_inter = UtilFunctions().radius_loss_interclass(latent_proto, latent_radius) * self.beta
                    radius_loss_intra = UtilFunctions().radius_loss_intraclass(latent_proto, latent_radius, self.shots, self.ways) * self.alpha
                

                    train_loss = recon_loss + radius_loss_inter + radius_loss_intra
                    train_loss /= len(adaptation_indices)

                    # Fast adaptation step in base model 
                    img_gen_learner.adapt(train_loss)

                # Query set as input to model 
                query_recon, latent_proto, latent_radius, latent_feat = img_gen_learner(query_set, gamma = 1.0)
               
                recon_loss = UtilFunctions().loss_reconstruction(query_set, query_recon)
                radius_loss_inter = UtilFunctions().radius_loss_interclass(latent_proto, latent_radius) * self.beta
                radius_loss_intra = UtilFunctions().radius_loss_intraclass(latent_proto, latent_radius, self.shots, self.ways) * self.alpha

                query_loss = recon_loss + radius_loss_inter + radius_loss_intra
                query_loss /= len(query_set)

                # Backpropagation of the meta model 
                query_loss.backward()
                meta_loss += query_loss.item()

            meta_loss /= self.meta_train_batch
            for params in img_gen_meta.parameters():
                params.grad.data.mul_(1.0 / self.meta_train_batch)
            opti_meta.step()

            if (i+1) % 100 == 0 or i == 0:
                print(i + 1, " Meta Train Loss: ", meta_loss)


        # Model path and model saving step 
        MODEL_PATH = os.getcwd() + '/img_gen.pt'
        torch.save(img_gen.state_dict(), MODEL_PATH)

        return MODEL_PATH
