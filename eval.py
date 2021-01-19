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
import time

from model import ImageGenerator
from utils import UtilFunctions


class ModelEvaluation():
    def __init__(self, test_tasks, shots, ways, base_lr, alpha, beta, meta_test_batch, meta_test_epochs, steps_per_task, latent_dims, device):
        self.test_tasks = test_tasks
        self.shots = shots
        self.ways = ways
        self.base_lr = base_lr
        self.alpha = alpha
        self.beta = beta
        self.meta_test_batch = meta_test_batch
        self.meta_test_epochs = meta_test_epochs
        self.steps_per_task = steps_per_task
        self.latent_dims = latent_dims
        self.device = device

    def eval(self, MODEL_PATH):
        img_gen = ImageGenerator(self.latent_dims, self.ways, self.shots)
        img_gen.load_state_dict(torch.load(MODEL_PATH))
        img_gen.to(self.device)
        img_gen_meta = l2l.algorithms.MAML(img_gen, lr = self.base_lr, first_order = True)

        print("Stauts: Evaluaton Stage \n")

        avg_recon_loss = 0.0
        total_fid = 0.0
        for i in range(self.meta_test_epochs):

            query_recon_loss = 0.0
            for t_idx in range(self.meta_test_batch):

                img_gen_learner = img_gen_meta.clone()
                test_task = self.test_tasks.sample()
                data, labels = test_task
                data = data.to(self.device)
                labels = labels.to(self.device)

                # img_gen_learner.train()

                adaptation_indices = np.zeros(data.size(0), dtype=bool)
                adaptation_indices[np.arange(self.shots * self.ways)*2] = True
                evaluation_indices = torch.from_numpy(~adaptation_indices).to(self.device)
                adaptation_indices = torch.from_numpy(adaptation_indices).to(self.device)

                support_set, support_labels = data[adaptation_indices], labels[adaptation_indices]
                query_set, query_labels = data[evaluation_indices], labels[evaluation_indices]
                recon_loss_list = []
                inter_loss_list = []
                intra_loss_list = []

                print("Reconstruction \t Inter-Clustering Loss \t Intra-Clustering Loss\n")
                for step in range(self.steps_per_task):
                    support_recon, latent_proto, latent_radius, latent_feat = img_gen_learner(support_set, gamma = 1.0)

                    recon_loss = UtilFunctions().loss_reconstruction(support_set, support_recon)
                    inter_clustering_loss = UtilFunctions().radius_loss_interclass(latent_proto, latent_radius) * self.beta
                    intra_clustering_loss = UtilFunctions().radius_loss_intraclass(latent_proto, latent_radius, self.shots, self.ways) * self.alpha
                    recon_loss_list.append(recon_loss.item())
                    inter_loss_list.append(inter_clustering_loss.item())
                    intra_loss_list.append(intra_clustering_loss.item())

                    train_loss = recon_loss + inter_clustering_loss + intra_clustering_loss 
                    train_loss /= len(adaptation_indices)
                    img_gen_learner.adapt(train_loss)
                    print("Step  ", step+1, " ", recon_loss.item(), " ", inter_clustering_loss.item(), " ", intra_clustering_loss.item())

                # img_gen_learner.eval()
                query_recon, latent_proto, latent_radius, latent_feat = img_gen_learner(query_set, gamma = 1.0)
                # query_recon_1, latent_proto_1, latent_radius_1, latent_feat_1 = img_gen_learner(query_set, gamma = 0.8)
                # query_recon_2, latent_proto_2, latent_radius_2, latent_feat_2 = img_gen_learner(query_set, gamma = 0.2)
                # query_recon_3, latent_proto_3, latent_radius_3, latent_feat_3 = img_gen_learner(query_set, gamma = -0.2)
                # query_recon_4, latent_proto_4, latent_radius_4, latent_feat_4 = img_gen_learner(query_set, gamma = -0.8)
                  

                recon_loss = UtilFunctions().loss_reconstruction(query_set, query_recon)
                inter_clustering_loss = UtilFunctions().radius_loss_interclass(latent_proto, latent_radius) * self.beta
                intra_clustering_loss = UtilFunctions().radius_loss_intraclass(latent_proto, latent_radius, self.shots, self.ways) * self.alpha

                # print("Task No. ", t_idx+1, "\t", recon_loss.item(), "\t", inter_clustering_loss.item(), "\t", intra_clustering_loss.item(), "\n" )

                query_loss = recon_loss + inter_clustering_loss + intra_clustering_loss
                query_loss /= len(query_set)
                query_recon_loss += query_loss

                # Generated Image Visualization
                # UtilFunctions().image_visual(query_set, query_recon, self.shots)
             
                # Generated Point Visualization
                # UtilFunctions().class_distribution_visual(latent_proto, latent_radius)
              
                # Plot of the Loss Functions 
                UtilFunctions().loss_plot(recon_loss_list, inter_loss_list, intra_loss_list)

                # FID score Estimation
                fid_score = UtilFunctions().calculate_fid(query_set, query_recon, self.shots)
              

                total_fid += fid_score
                print("FID ", fid_score)

            # query_recon_loss /= self.meta_test_batch
            print(i+1, " Query Reconstruction Loss: ", query_recon_loss.item())

            avg_recon_loss += query_recon_loss

        avg_recon_loss = avg_recon_loss / self.meta_test_epochs
        print("\nAverage Reconstruction Loss:  ", avg_recon_loss.item())
        print("Average FID score  ", (total_fid / self.meta_test_epochs))
