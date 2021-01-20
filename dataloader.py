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


# Data loader for MNIST dataset
class MNISTLoader():
	def __init__(self, nways, shots, query, num_train_tasks, num_test_tasks):
		print("Dataset:  MNIST\n")
		self.nways = nways
		self.shots = shots
		self.query = query
		self.num_train_tasks = num_train_tasks
		self.num_test_tasks = num_test_tasks

		total_classes = 10
		class_set = [i for i in range(total_classes)]
		# self.train_classes = [0, 1, 2, 3, 4, 5, 6, 7, 8]
		self.train_classes = random.sample(class_set, total_classes - self.nways)
		self.test_classes = list(set(class_set).difference(self.train_classes))

		# self.transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
		self.transforms = transforms.Compose([transforms.ToTensor()])


	def train_task_set(self):
		mnist_train = l2l.data.MetaDataset(torchvision.datasets.MNIST(root='./data',
                                         train=True,
                                         download=True,
                                         transform=self.transforms))

		train_tasks = l2l.data.TaskDataset(mnist_train,
                                   task_transforms=[
                                        l2l.data.transforms.FusedNWaysKShots(mnist_train, n = self.nways, k = (self.shots + self.query),
                                        									replacement = False, filter_labels = self.train_classes),
                                        l2l.data.transforms.LoadData(mnist_train),
                                        l2l.data.transforms.RemapLabels(mnist_train),
                                        l2l.data.transforms.ConsecutiveLabels(mnist_train),
                                   ],
                                   num_tasks = self.num_train_tasks)

		return train_tasks


	def test_task_set(self):
		mnist_test = l2l.data.MetaDataset(torchvision.datasets.MNIST(root='./data',
                                         train=False,
                                         download=True,
                                         transform=self.transforms))

		test_tasks = l2l.data.TaskDataset(mnist_test,
                                   task_transforms=[
                                        l2l.data.transforms.FusedNWaysKShots(mnist_test, n = self.nways, k = (self.shots + self.query),
                                        									replacement = False, filter_labels = self.test_classes),
                                        l2l.data.transforms.LoadData(mnist_test),
                                        l2l.data.transforms.RemapLabels(mnist_test),
                                        l2l.data.transforms.ConsecutiveLabels(mnist_test),
                                   ],
                                   num_tasks = self.num_test_tasks)

		return test_tasks

# Data loader for fashion-MNIST dataset
class Fashion_MNISTLoader():
	def __init__(self, nways, shots, query, num_train_tasks, num_test_tasks):
		print("Dataset:  Fashion MNIST\n")
		self.nways = nways
		self.shots = shots
		self.query = query
		self.num_train_tasks = num_train_tasks
		self.num_test_tasks = num_test_tasks

		total_classes = 10
		class_set = [i for i in range(total_classes)]
		self.train_classes = random.sample(class_set, total_classes - self.nways)
		self.test_classes = list(set(class_set).difference(self.train_classes))

		# self.transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
		self.transforms = transforms.Compose([transforms.ToTensor()])


	def train_task_set(self):
		mnist_train = l2l.data.MetaDataset(torchvision.datasets.FashionMNIST(root='./data',
                                         train=True,
                                         download=True,
                                         transform=self.transforms))

		train_tasks = l2l.data.TaskDataset(mnist_train,
                                   task_transforms=[
                                        l2l.data.transforms.FusedNWaysKShots(mnist_train, n = self.nways, k = (self.shots + self.query),
                                        									replacement = False, filter_labels = self.train_classes),
                                        l2l.data.transforms.LoadData(mnist_train),
                                        l2l.data.transforms.RemapLabels(mnist_train),
                                        l2l.data.transforms.ConsecutiveLabels(mnist_train),
                                   ],
                                   num_tasks = self.num_train_tasks)

		return train_tasks


	def test_task_set(self):
		mnist_test = l2l.data.MetaDataset(torchvision.datasets.FashionMNIST(root='./data',
                                         train=False,
                                         download=True,
                                         transform=self.transforms))

		test_tasks = l2l.data.TaskDataset(mnist_test,
                                   task_transforms=[
                                        l2l.data.transforms.FusedNWaysKShots(mnist_test, n = self.nways, k = (self.shots + self.query),
                                        									replacement = False, filter_labels = self.test_classes),
                                        l2l.data.transforms.LoadData(mnist_test),
                                        l2l.data.transforms.RemapLabels(mnist_test),
                                        l2l.data.transforms.ConsecutiveLabels(mnist_test),
                                   ],
                                   num_tasks = self.num_test_tasks)

		return test_tasks

# Data loader for Omniglot dataset
class OmniglotLoader():
	def __init__(self, nways, shots, query, num_train_tasks, num_test_tasks):
		print("Dataset:  Omniglot\n")
		self.nways = nways
		self.shots = shots
		self.query = query
		self.num_train_tasks = num_train_tasks
		self.num_test_tasks = num_test_tasks

		total_classes = 1623
		class_set = [i for i in range(total_classes)]
		self.train_classes = random.sample(class_set, 1200)
		self.test_classes = list(set(class_set).difference(self.train_classes))

		self.transforms = transforms.Compose([transforms.Resize((28, 28)), transforms.ToTensor()])


	def train_task_set(self):
		background_set = l2l.data.MetaDataset(torchvision.datasets.Omniglot(root='./data',
                                         background=True,
                                         download=True,
                                         transform=self.transforms))

		train_tasks = l2l.data.TaskDataset(background_set,
                                   task_transforms=[
                                        l2l.data.transforms.FusedNWaysKShots(background_set, n = self.nways, k = (self.shots + self.query),
                                        									replacement = False),
                                        l2l.data.transforms.LoadData(background_set),
                                        l2l.data.transforms.RemapLabels(background_set),
                                        l2l.data.transforms.ConsecutiveLabels(background_set),
                                   ],
                                   num_tasks = self.num_train_tasks)

		return train_tasks


	def test_task_set(self):
		evaluation_set = l2l.data.MetaDataset(torchvision.datasets.Omniglot(root='./data',
                                         background=False,
                                         download=True,
                                         transform=self.transforms))

		test_tasks = l2l.data.TaskDataset(evaluation_set,
                                   task_transforms=[
                                        l2l.data.transforms.FusedNWaysKShots(evaluation_set, n = self.nways, k = (self.shots + self.query),
                                        									replacement = False),
                                        l2l.data.transforms.LoadData(evaluation_set),
                                        l2l.data.transforms.RemapLabels(evaluation_set),
                                        l2l.data.transforms.ConsecutiveLabels(evaluation_set),
                                   ],
                                   num_tasks = self.num_test_tasks)

		return test_tasks


