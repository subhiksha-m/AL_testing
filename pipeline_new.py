import shutil
import torch
import requests
from IPython import get_ipython
from tqdm.notebook import tqdm
from torchvision import transforms
import os
import pathlib
from imutils import paths
import json
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
import time
import yaml
import random
import torchvision
from torchvision import transforms

##sim search
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
import sys

import logging

logging.info("APP START")

sys.path.insert(0, "Self-Supervised-Learner")
sys.path.insert(1, "ActiveLabelerModels")
sys.path.insert(1, "ActiveLabeler-main")
from models import CLASSIFIER
from models import SIMCLR
from models import SIMSIAM
from ActiveLabeler import ActiveLabeler
from TrainModels import TrainModels
#from SimilaritySearch import SimilaritySearch
#from Diversity_Algorithm.diversity_func import Diversity


class Pipeline:
    def __init__(self, config_path):
        self.config_path = config_path
        self.dataset_paths=[]
        self.unlabeled_list = []
        self.labled_list = []
        self.embeddings = None
        self.div_embeddings = None

    # similiarity search class
    def get_annoy_tree(self, num_nodes, embeddings, num_trees, annoy_path):
        t = AnnoyIndex(num_nodes, 'euclidean')
        for i in range(len(embeddings)):
            t.add_item(i, embeddings[i])
        t.build(num_trees)
        t.save(annoy_path)
        print("Annoy file stored at ", annoy_path)

    def inference(self, image_path, model):
        im = Image.open(image_path).convert('RGB')
        image = np.transpose(im, (2, 0, 1)).copy()
        im = torch.tensor(image).unsqueeze(0).float().cuda()
        x = model(im)
        return x[0]

    def get_nn_annoy(self, image_path, n_closest, num_nodes, annoy_path, data_path, disp=False):
        # load dependencies
        u = AnnoyIndex(num_nodes, 'euclidean')
        u.load(annoy_path)

        # get image embedding
        image_embedding = self.inference(image_path)

        dataset_paths = list(paths.list_images(data_path))

        inds, dists = u.get_nns_by_vector(image_embedding, n_closest, include_distances=True)

        if disp:
            for i in range(len(inds)):
                print("Class:", dataset_paths[inds[i]].split("/")[-2])
                print("Distance:", dists[i])

        return inds, dists

    def generate_embeddings(self, image_size, embedding_size, model, dataset_paths):
        t = transforms.Resize((image_size, image_size))
        embedding_matrix = torch.empty(size=(0, embedding_size)).cuda()
        model = model
        # model.eval()
        # model.cuda()
        for f in tqdm(dataset_paths):
            with torch.no_grad():
                im = Image.open(f).convert('RGB')
                im = t(im)
                im = np.asarray(im).transpose(2, 0, 1)
                im = torch.Tensor(im).unsqueeze(0).cuda()
                #embedding = model(im)[0]
                embedding = model.encoder(im)[-1]
                embedding_matrix = torch.vstack((embedding_matrix, embedding))
        logging.info(f'Got embeddings. Embedding Shape: {embedding_matrix.shape}')
        print(f'\nGot embeddings. Embedding Shape: {embedding_matrix.shape}')
        return embedding_matrix.detach().cpu().numpy()

    def initialize_embeddings(self, image_size, embedding_size, model, dataset_paths, num_nodes, num_trees, annoy_path):
        self.embeddings = self.generate_embeddings(image_size, embedding_size, model, dataset_paths)
        self.get_annoy_tree(self, num_nodes, self.embeddings, num_trees, annoy_path)


    def search_similar(self, ref_imgs, n_closest):
        image_names = set()
        for image_path in ref_imgs:
            inds, dists = self.get_nn_annoy(image_path, n_closest)
            for idx in inds:
                image_names.add(self.dataset_paths[idx])
        image_names = list(image_names)
        return image_names

    def label_data(self, image_paths_or_names, data_path, swipe_url,simulate_label=False,unlabeled_path=None, labeled_path=None, postive_path=None,
                   negative_path=None, unsure_path=None):

        logging.info("Deduplicate and prepare for labeling")

        image_names = [image_path.split("/")[-1] for image_path in image_paths_or_names]
        image_names = list(set(image_names))
        for labled in self.labled_list:
            if labled in image_names:
                image_names.remove(labled)

        for image_name in image_names:
            image_path_copy = data_path + "/Unlabeled/" + image_name  # TODO TODO search in all
            shutil.copy(image_path_copy, unlabeled_path)
            self.unlabeled_list.remove(image_name)  # pop and store if unlabeled paths needed
            self.labled_list.append(image_name)

        logging.debug(f"images sent to labeling: {image_names}")
        self.swipe_label_simulate(swipe_url=swipe_url,simulate_label=simulate_label,unlabled_path=unlabeled_path, labeled_path=labeled_path, positive_path=postive_path,
                                  negative_path=negative_path, unsure_path=unsure_path)

    def swipe_label_simulate(self, swipe_url, simulate_label=False,unlabled_path=None, labeled_path=None, positive_path=None, negative_path=None,
                             unsure_path=None):
        logging.info("Calling swipe labeler")
        print(f'\n {len(list(paths.list_images(unlabled_path)))} images to label. Go to {swipe_url}')

        ori_labled = len(list(paths.list_images(labeled_path)))
        ori_pos = len(list(paths.list_images(positive_path)))
        ori_neg = len(list(paths.list_images(negative_path)))

        if simulate_label: # TODO
            for img in list(paths.list_images(unlabled_path)):
                src = unlabled_path + "/" + img.split('/')[-1]
                dest = (positive_path + "/" + img.split('/')[-1]) if "airplane" in img else (
                        negative_path + "/" + img.split('/')[-1])
                shutil.move(src, dest)

        # else:
        #     # HTML(f"<a href={self.swipe_url.public_url}>\n{len(list(paths.list_images(unlabled_path)))} images to label</a>")
        #
        #     # batch_size = (len(list(paths.list_images(unlabled_path))) // 2) + 1
        #     batch_size = min(len(list(paths.list_images(unlabled_path))), self.swipe_label_batch_size)
        #     # TODO: get from user ? depending on number of users of link ?
        #     tic = time.perf_counter()
        #     label = f"python3 {self.swipe_dir}/Swipe-Labeler-main/api/api.py --path_for_unlabeled='{unlabled_path}' --path_for_pos_labels='{positive_path}' --path_for_neg_labels='{negative_path}' --path_for_unsure_labels='{unsure_path}' --batch_size={batch_size} > swipelog"
        #     # >/dev/null 2>&1"
        #     logging.debug(label)
        #     os.system(label)
        #     toc = time.perf_counter()
        #     self.human_time += toc - tic  # seconds

        print(
            f' {len(list(paths.list_images(labeled_path))) - ori_labled} labeled: {len(list(paths.list_images(positive_path))) - ori_pos} Pos {len(list(paths.list_images(negative_path))) - ori_neg} Neg')

        logging.info(
            f'{len(list(paths.list_images(labeled_path)))} labeled: {len(list(paths.list_images(positive_path)))} Pos {len(list(paths.list_images(negative_path)))} Neg')
        logging.info(f"unlabeled list: {self.unlabeled_list}")
        logging.info(f"labeled list: {self.labled_list}")

    def create_seed_dataset(self, ref_img,data_path,swipe_url,simulate_label,unlabled_path=None, labeled_path=None, positive_path=None, negative_path=None,
                             unsure_path=None):
        iteration = 0
        n_closest = 1
        while True:
            iteration += 1
            print(f'\n----- iteration: {iteration}')

            print("Enter n closest")
            n_closest = input()
            if n_closest == 0: exit()

            ref_imgs = [ref_img] if iteration == 1 else paths.list_images(positive_path)
            imgs = self.search_similar(ref_imgs, (n_closest * 8) // 10)

            # random sampling 80:20
            #n_20 = n_closest - n_closest * 8 // 10
            n_20 = len(imgs)//4
            r_imgs = random.choices((self.unlabeled_list-imgs), k=n_20)
            imgs = imgs + r_imgs

            self.label_data(imgs, data_path=data_path,swipe_url=swipe_url,simulate_label=simulate_label, unlabled_path=unlabled_path, labeled_path=labeled_path, positive_path=positive_path,
                                  negative_path=negative_path, unsure_path=unsure_path)

    def load_config(self, config_path):
        with open(config_path) as file:
            config = yaml.safe_load(file)
        return config

    def load_model(self, model_type, model_path, data_path):  # , device):
        # TODO loading simsiam vs simclr models
        # TODO device
        if model_type == "simclr":
            model = SIMCLR.SIMCLR.load_from_checkpoint(model_path, DATA_PATH=data_path)
            logging.info("simclr model loaded")
        else:
            model = SIMSIAM.SIMSIAM.load_from_checkpoint(model_path, DATA_PATH=data_path)
            logging.info("simsiam model loaded")
        model.to('cuda')
        model.eval()
        return model

    def create_emb_label_mapping(self,img_names,positive_path,negative_path):
        # emb_dataset = [[emb,label]..] 0-neg, 1 -pos
        emb_dataset= []
        pos_label = list(paths.list_images(positive_path))
        neg_label = list(paths.list_images(negative_path))
        i = -1
        for img_path in self.dataset_paths:
            i=i+1
            if img_path.split("/")[-1] in img_names:
                label = 1 if img_path.split("/")[-1] in pos_label else 0
                emb_dataset.append([self.embeddings[i],label])
        return  emb_dataset

    def create_emb_mapping(self,img_names):
        # emb_dataset = [[emb,label]..] 0-neg, 1 -pos
        emb_dataset= []
        i = -1
        for img_path in self.dataset_paths:
            i=i+1
            if img_path.split("/")[-1] in img_names:
                emb_dataset.append(self.embeddings[i])
        return  emb_dataset

    def main(self):
        # offline

        logging.info('load config')
        parameters = self.load_config(self.config_path)

        logging.info('load model')
        model = self.load_model(parameters['model']['model_type'], parameters['model']['model_path'], parameters['data']['data_path'])

        logging.info('initialize_embeddings')
        self.initialize_embeddings(parameters['model']['image_size'], parameters['model']['embedding_size'], model,
                                   list(paths.list_images(parameters['data']['data_path'])), parameters['annoy']['num_nodes'],
                                   parameters['annoy']['num_trees'], parameters['annoy']['annoy_path'])

        logging.info('create_seed_dataset')
        self.dataset_paths = list(paths.list_images(parameters['data']['data_path']))
        self.unlabeled_list = [i.split('/')[-1] for i in self.dataset_paths]
        self.labled_list = []
        self.create_seed_dataset(parameters['nn']['ref_img_path'],parameters['data']['data_path'],parameters['nn']['swipe_url'],parameters['nn']['simulate_label'],
                                 parameters['nn']['unlabled_path'],parameters['nn']['labeled_path'],parameters['nn']['positive_path'],parameters['nn']['negative_path'],parameters['nn']['unsure_path'])

        logging.info('active_labeling')
        logging.info("Initializing active labeler and diversity algorithm class objects.")
        #whatever unlabled images left
        #has to be updated when not using diversity and using entire dataset
        activelabeler = ActiveLabeler(self.create_emb_mapping(self.unlabeled_list), list(paths.list_images(self.unlabeled_list))
        #dummy dataset - 3/4 #TODO Create
        train_models = TrainModels(parameters['TrainModels']['config_path'],"./",parameters['TrainModels']['dummy_dataset'], "AL") #TODO saved model path # datapath => sub directory structure for datapath arg
        #TODO diversity

        def to_tensor(pil):
            return torch.tensor(np.array(pil)).permute(2, 0, 1).float()

        t = transforms.Compose([
            transforms.Resize((parameters['model']['image_size'], parameters['model']['image_size'])),
            transforms.Lambda(to_tensor)
        ])

        iteration= 0
        newly_labled_path = parameters['nn']['labeled_path']
        while True:
            iteration +=1
            print(f"iteration {iteration}")
            # entire model - image_dataloader
            # offline, train_linear

            #label - emb
            # lists emb , labels - 0(neg),1(pos) , labled image names - not req AL => mapped via indices
            #emb_dataset = [[emb,label]..]
            #newly_labeld
            #sample 80:20
            emb_dataset = self.create_emb_label_mapping(newly_labled_path,parameters['nn']['positive_path'],parameters['nn']['negative_path'])
            emb_dataset = random.sample(emb_dataset)
            n_80 = (len(emb_dataset)*8)//10
            training_dataset = DataLoader(emb_dataset[:n_80], batch_size = 32) #TODO yml
            validation_dataset = DataLoader(emb_dataset[n_80+1:], batch_size = 1)
            train_models.train_linear(training_dataset, validation_dataset)
            #{self.model_path}AL_0
            strategy_embeddings, strategy_images= activelabeler.get_images_to_label_offline(train_models.get_model(), "uncertainty", parameters['ActiveLabeler']['sample_size'], None, "cuda")

            #nn for each emb
            # label
            #archive => pos , neg
            imgs = self.search_similar(strategy_images, parameters['AL_main']['n_closest'])
            imgs = imgs + strategy_images

            self.label_data(imgs, data_path=parameters['data']['data_path'], swipe_url=parameters['nn']['swipe_url'], simulate_label=parameters['nn']['simulate_label'],
                            unlabled_path=parameters['nn']['unlabled_path'], labeled_path=parameters['AL_main']['newly_labled_path'], positive_path=parameters['AL_main']['newly_labled_path']+"/positive",
                            negative_path=parameters['AL_main']['newly_labled_path']+"/negative", unsure_path=None)
            for img in list(paths.list_images(newly_labled_path)):
                shutil.copy(img,parameters['AL_main']['archive_path'])
            newly_labled_path = parameters['AL_main']['newly_labled_path']
            for img in list(paths.list_images(newly_labled_path)):
                os.remove(img)

            activelabeler.get_embeddings_offline(self.create_emb_mapping(self.unlabeled_list), list(paths.list_images(self.unlabeled_list))

            print("Enter c to continue")
            input_counter = input()
            if input_counter != 'c': exit()

            #only for offline
            #TODO model_confident
            if iteration>3:
                #  archive
                # sample 80:20 #TODO
                logging.info("model_confident")
                archive_dataset = torchvision.datasets.ImageFolder("parameters['AL_main']['archive_path']", t)
                training_dataset, validation_dataset= torch.utils.data.random_split(archive_dataset,[int(0.8*(len(archive_dataset))),int(0.2*(len(archive_dataset)))])
                training_dataset = DataLoader(training_dataset, batch_size = 32)
                validation_dataset = DataLoader(validation_dataset, batch_size = 1)
                train_models.train_all(training_dataset, validation_dataset)
                #self.initialize_embeddings(image_size, embedding_size, train_models.get_model(), dataset_paths, num_nodes, num_trees, annoy_path)
                #forward pass - get_images_to_label_offline()
                #label
                #linear layer


