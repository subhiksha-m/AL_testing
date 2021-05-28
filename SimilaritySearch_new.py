from torch.utils.data import DataLoader
from annoy import AnnoyIndex
from torchvision import transforms
from argparse import ArgumentParser

import torchvision.datasets as datasets
import torch
import pickle
import os
from tqdm.notebook import tqdm
from torchvision import transforms

import shutil

import PIL.Image as Image
import torch
import numpy as np
import matplotlib.pyplot as plt
from annoy import AnnoyIndex
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.image as mpimg
from imutils import paths


class SimilaritySearch:
    def __init__(self, image_size, data_path, dataset_paths, model, annoy_path, embedding_size, embeddings, device,num_trees,
                 batch_size):
        self.image_size = image_size
        self.data_path = data_path
        self.dataset_paths = dataset_paths
        self.model = model
        self.annoy_path = annoy_path
        self.embedding_size = embedding_size
        self.embeddings = embeddings
        self.device = device
        self.num_nodes = embedding_size
        self.num_trees = num_trees
        self.batch_size = batch_size
        # self.image_path = image_path
        # self.image_embedding = self.inference()
        # self.n_closest = n_closest

    def get_annoy_tree(self):
        t = AnnoyIndex(self.num_nodes, 'euclidean')
        for i in range(len(self.embeddings)):
            t.add_item(i, self.embeddings[i])
        t.build(self.num_trees)
        t.save(self.annoy_path)
        print("Annoy file stored at ", self.annoy_path)

    def inference(self, image_path):
        im = Image.open(image_path).convert('RGB')
        image = np.transpose(im, (2, 0, 1)).copy()
        im = torch.tensor(image).unsqueeze(0).float().cuda()
        x = self.model(im)
        return x[0]

    def get_nn_annoy(self, image_path, n_closest, disp=False):
        # load dependencies
        u = AnnoyIndex(self.num_nodes, 'euclidean')
        u.load(self.annoy_path)

        # get image embedding
        image_embedding = self.inference(image_path)

        dataset_paths = list(paths.list_images(self.data_path))

        inds, dists = u.get_nns_by_vector(image_embedding, n_closest, include_distances=True)

        if disp:
            for i in range(len(inds)):
                print("Class:", dataset_paths[inds[i]].split("/")[-2])
                print("Distance:", dists[i])

        return inds, dists
