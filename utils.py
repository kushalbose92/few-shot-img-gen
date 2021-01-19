import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
import torchvision
from torch.autograd import Variable
import random
from math import pi
from sklearn.manifold import TSNE
from sklearn import preprocessing
import datetime
import os
import copy
import torchvision.transforms as transforms
import learn2learn as l2l
from numpy import cov
from numpy import trace
from numpy import iscomplexobj
# from numpy.random import random
from scipy.linalg import sqrtm

from model import ImageGenerator

bce_loss = nn.BCELoss(reduction = 'sum')

class UtilFunctions():
    def __init__(self):
        return

    def image_visual(self, query_set, query_recon, shots):
        full_set = torch.cat([query_set, query_recon], 0)
        full_set = 1 - full_set.clamp(0, 1)
        np_imagegrid = torchvision.utils.make_grid(full_set, shots, 2).cpu().detach().numpy()
        plt.imshow(np.transpose(np_imagegrid, (1, 2, 0)))
        path = os.getcwd()
        now = datetime.datetime.now()
        timestamp = str(now.strftime("%Y%m%d_%H:%M:%S"))
        plt.savefig(path + "/results_fashion_mnist/generated_images_" + timestamp + "_.png")
        # plt.clf()

    def calculate_fid(self, act1, act2, shots):
        act1 = act1.reshape(shots, -1)
        act2 = act2.reshape(shots, -1)
        act1 = act1.detach().cpu().numpy()
        act2 = act2.detach().cpu().numpy()
        # calculate mean and covariance statistics
        mu1, sigma1 = act1.mean(axis=0), cov(act1, rowvar=False)
        mu2, sigma2 = act2.mean(axis=0), cov(act2, rowvar=False)
        # calculate sum squared difference between means
        ssdiff = np.sum((mu1 - mu2)**2.0)
        # calculate sqrt of product between cov
        t = sigma1.dot(sigma2)
        # print(t)
        covmean = sqrtm(t)
        # check and correct imaginary numbers from sqrt
        if iscomplexobj(covmean):
            covmean = covmean.real
        # calculate score
        fid = ssdiff + trace(sigma1 + sigma2 - 2*covmean)
        return fid

    def class_distribution_visual(self, latent_proto, latent_radius):

        gen_points = torch.add(latent_proto, latent_radius)
        gen_points_1 = torch.add(latent_proto, latent_radius * 0.70)
        gen_points_2 = torch.add(latent_proto, latent_radius * 0.30)
        gen_points_3 = torch.add(latent_proto, latent_radius * 0.70 * (-1))
        gen_points_4 = torch.add(latent_proto, latent_radius * 0.30 * (-1))
        latent_proto = latent_proto.cpu().detach().numpy()
        latent_radius = latent_radius.cpu().detach().numpy()
        # print("proto  ", latent_proto)
        # print("radius  ", latent_radius)
        gen_points = gen_points.cpu().detach().numpy()
        gen_points_1 = gen_points_1.cpu().detach().numpy()
        gen_points_2 = gen_points_2.cpu().detach().numpy()
        gen_points_3 = gen_points_3.cpu().detach().numpy()
        gen_points_4 = gen_points_4.cpu().detach().numpy()
        num_points = len(latent_proto)
        t = np.linspace(0, 2*pi, 50)
        for s_idx in range(num_points):

            plt.plot(latent_proto[s_idx][0] + latent_radius[s_idx][0]*np.cos(t), latent_proto[s_idx][1] + latent_radius[s_idx][1]*np.sin(t))
            plt.plot(latent_proto[s_idx][0], latent_proto[s_idx][1], "ro")
            # plt.plot(gen_points[s_idx][0], gen_points[s_idx][1], "b^")
            plt.plot(gen_points_1[s_idx][0], gen_points_1[s_idx][1], "b^")
            plt.plot(gen_points_2[s_idx][0], gen_points_2[s_idx][1], "b^")
            plt.plot(gen_points_3[s_idx][0], gen_points_3[s_idx][1], "b^")
            plt.plot(gen_points_4[s_idx][0], gen_points_4[s_idx][1], "b^")

        path = os.getcwd()
        now = datetime.datetime.now()
        timestamp = str(now.strftime("%Y%m%d_%H:%M:%S"))
        plt.savefig(path + "/results_class/generated_classes_" + timestamp + "_.png")
        plt.clf()

    def loss_plot(self, recon_loss_list, inter_loss_list, intra_loss_list):
        path = os.getcwd()
        plt.plot(recon_loss_list, label='Reconstruction Loss')
        plt.plot(inter_loss_list, label='Inter-Clustering Loss')
        plt.plot(intra_loss_list, label='Intra-Clustering Loss')
        now = datetime.datetime.now()
        timestamp = str(now.strftime("%Y%m%d_%H:%M:%S"))
        plt.xlabel("Iterations")
        plt.ylabel("Loss Value")
        plt.title("Plot of Loss Functions")
        plt.legend()
        plt.savefig(path + "/loss_plot/images_" + timestamp +"_.png")
        plt.clf()

    # Loss to be Minimized
    def radius_loss_intraclass(self, x_proto, x_radius, shots, ways):
       
        intra_loss = 0.0
        # print(x_latent.shape)

        for i in range(ways):

            intra_loss += torch.norm(x_radius[i], p=2)
        # for w in range(ways):
        #     for s in range(shots):
        #         intra_loss += torch.norm(torch.sub(x_proto[w], x_latent[ways*w + s]), p = 2)

        return intra_loss

    # Loss to be Maximized
    # Loss between pair of generated points from two different class
    def radius_loss_interclass(self, x_proto, x_radius):
       
        x_generated = torch.add(x_proto, x_radius)
        num_points = x_proto.size(0)
        latent_dims = x_proto.size(1)
        size = int((num_points)*(num_points))
        inter_loss = 0.0

        for i in range(num_points):
            for j in range(num_points):
               inter_loss += torch.norm(torch.sub(x_generated[i], x_generated[j]), p = 2)

        return -1 * inter_loss

    def loss_reconstruction(self, input_images, generated_images):
        recon_loss = F.binary_cross_entropy(generated_images.view(-1, 1*28*28), input_images.view(-1, 1*28*28), reduction='sum')
        # recon_loss = F.mse_loss(generated_images, input_images, reduction = 'sum')
        return recon_loss 
