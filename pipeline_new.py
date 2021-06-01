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
from sys import exit

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
random.seed(100)

sys.path.insert(0, "Self-Supervised-Learner")
sys.path.insert(0, "ActiveLabelerModels")
sys.path.insert(0, "ActiveLabeler-main")
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
        self.initialize_emb_counter =0

    # similiarity search class
    def get_annoy_tree(self, num_nodes, embeddings, num_trees, annoy_path):
        t = AnnoyIndex(num_nodes, 'euclidean')
        for i in range(len(embeddings)):
            t.add_item(i, embeddings[i])
        t.build(num_trees)
        t.save(annoy_path)
        print("Annoy file stored at ", annoy_path)

    def inference(self, image_path, model,model_type):
        im = Image.open(image_path).convert('RGB')
        image = np.transpose(im, (2, 0, 1)).copy()
        im = torch.tensor(image).unsqueeze(0).float().cuda()
        if model_type == "model":
            x = model(im)
            x=x[0]
        else:
            x = model(im)[-1]
        return x


    def get_nn_annoy(self, image_path, n_closest, num_nodes, annoy_path, data_path,model, model_type,disp=False):
        # load dependencies
        u = AnnoyIndex(num_nodes, 'euclidean')
        u.load(annoy_path)

        # if emb not passed use inference function and model to generate emb
        #if image_embedding is None:
        image_embedding = self.inference(image_path,model,model_type) #TODO embedding

        #dataset_paths = list(paths.list_images(data_path))

        inds, dists = u.get_nns_by_vector(image_embedding, n_closest, include_distances=True)

        # if disp:
        #     for i in range(len(inds)):
        #         print("Class:", dataset_paths[inds[i]].split("/")[-2])
        #         print("Distance:", dists[i])

        return inds, dists

    def generate_embeddings(self, image_size, embedding_size, model, dataset_paths,model_type="model"):
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
                if model_type== "model":
                    embedding = model(im)[0]
                else: #encoder
                    embedding = model.encoder(im)[-1]
                embedding_matrix = torch.vstack((embedding_matrix, embedding))
        logging.info(f'Got embeddings. Embedding Shape: {embedding_matrix.shape}')
        print(f'\nGot embeddings. Embedding Shape: {embedding_matrix.shape}')
        return embedding_matrix.detach().cpu().numpy()


    def initialize_embeddings(self, image_size, embedding_size, model, dataset_paths, num_nodes, num_trees, annoy_path,model_type="model"):
        self.embeddings = self.generate_embeddings(image_size, embedding_size, model, dataset_paths,model_type)
        self.get_annoy_tree(embedding_size, self.embeddings, num_trees, annoy_path) #TODO numnodes = emb size

    def search_similar(self, ref_imgs, n_closest,num_nodes, annoy_path, data_path,model,model_type):
        image_names = set()
        for image_path in ref_imgs:
            inds, dists = self.get_nn_annoy(image_path, n_closest,num_nodes, annoy_path, data_path,model,model_type, False)
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

    # def get_embeddings(self,img_list):
    #     emb_dataset = []
    #     img_names = [i.split("/")[-1] for i in img_list]
    #
    #     i = -1
    #     for img_path in self.dataset_paths:
    #         i = i + 1
    #         if img_path.split("/")[-1] in img_names:
    #             emb_dataset.append(self.embeddings[i])
    #     return emb_dataset


    def create_seed_dataset(self, ref_img,data_path,swipe_url,simulate_label,num_nodes, annoy_path,model,unlabled_path=None, labeled_path=None, positive_path=None, negative_path=None,
                             unsure_path=None):
        iteration = 0
        n_closest = 1
        while True:
            iteration += 1
            print(f'\n----- iteration: {iteration}')

            print("Enter n closest")
            n_closest = input()
            n_closest = int(n_closest)
            if n_closest == 0: break

            ref_imgs = [ref_img] if iteration == 1 else list(paths.list_images(positive_path)) #self.get_embeddings(list(paths.list_images(positive_path)))
            imgs = self.search_similar(ref_imgs, (n_closest * 8) // 10,num_nodes, annoy_path, data_path,model,"model")

            # random sampling 80:20
            #n_20 = n_closest - n_closest * 8 // 10
            n_20 = len(imgs)//4
            tmp1 = set(self.unlabeled_list)
            tmp2 = set(imgs)
            tmp3 = list( tmp1-tmp2)
            r_imgs = random.choices(tmp3, k=n_20)
            imgs = imgs + r_imgs

            self.label_data(imgs, data_path,swipe_url,simulate_label, unlabled_path, labeled_path, positive_path,
                                  negative_path, unsure_path)

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

    def create_emb_label_mapping(self,positive_path,negative_path):
        # emb_dataset = [[emb,label]..] 0-neg, 1 -pos
        emb_dataset= []
        pos_label = [i.split("/")[-1] for i in list(paths.list_images(positive_path))]
        neg_label = [i.split("/")[-1] for i in list(paths.list_images(negative_path))]
        i = -1
        for img_path in self.dataset_paths:
            i=i+1
            if img_path.split("/")[-1] in pos_label:
                label = 1
                emb_dataset.append([self.embeddings[i], label])
            if img_path.split("/")[-1] in neg_label:
                label = 0
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

        # +++++++++++++++++++++++++++++++++++++++++++++++++++++
        #seed dataset

        logging.info('load config')
        parameters = self.load_config(self.config_path)

        logging.info('load model')
        model = self.load_model(parameters['model']['model_type'], parameters['model']['model_path'], parameters['data']['data_path'])

        logging.info('initialize_embeddings')
        self.initialize_emb_counter +=1
        parameters['annoy']['annoy_path'] = parameters['annoy']['annoy_path'] + f"{self.initialize_emb_counter}"
        self.initialize_embeddings(parameters['model']['image_size'], parameters['model']['embedding_size'], model,
                                   list(paths.list_images(parameters['data']['data_path'])), parameters['annoy']['num_nodes'],
                                   parameters['annoy']['num_trees'], parameters['annoy']['annoy_path'])

        logging.info('create_seed_dataset')
        self.dataset_paths = list(paths.list_images(parameters['data']['data_path']))
        self.unlabeled_list = [i.split('/')[-1] for i in self.dataset_paths]
        self.labled_list = []
        self.create_seed_dataset(parameters['nn']['ref_img_path'],parameters['data']['data_path'],parameters['nn']['swipe_url'],parameters['nn']['simulate_label'],parameters['annoy']['num_nodes'],parameters['annoy']['annoy_path'] ,model,
                                 parameters['nn']['unlabled_path'],parameters['nn']['labeled_path'],parameters['nn']['positive_path'],parameters['nn']['negative_path'],parameters['nn']['unsure_path'])

        os.chdir(parameters['AL_main']['al_folder'])
        logging.info('active_labeling')
        logging.info("Initializing active labeler and diversity algorithm class objects.")
        #whatever unlabled images left
        #has to be updated when not using diversity and using entire dataset
        activelabeler = ActiveLabeler(self.create_emb_mapping(self.unlabeled_list), [ parameters['data']['data_path'] + "/Unlabeled/" + image_name for image_name in self.unlabeled_list ] )
        #dummy dataset - 3/4 #TODO Create
        train_models = TrainModels(parameters['TrainModels']['config_path'],"./",parameters['TrainModels']['dummy_dataset'], "AL") #TODO saved model path # datapath => sub directory structure for datapath arg
        #TODO diversity

        def to_tensor(pil):
            return torch.tensor(np.array(pil)).permute(2, 0, 1).float()

        t = transforms.Compose([
            transforms.Resize((parameters['model']['image_size'], parameters['model']['image_size'])),
            transforms.Lambda(to_tensor)
        ])

        #+++++++++++++++++++++++++++++++++++++++++++++++++++++
        #AL new flow - combining linear model and finetuning

        iteration = 0
        model_type = "model"
        newly_labled_path = parameters['nn']['labeled_path']  #seed dataset
        while True:
            iteration += 1
            print(f"iteration {iteration}")

            print("Enter l for Linear, f for finetuning and q to quit")  # TODO unrecognizable char
            input_counter = input()
            if input_counter == 'q': break

            if input_counter == 'l':
                #linear = create newly labeled emb, sample and split , get uncertain
                emb_dataset = self.create_emb_label_mapping(newly_labled_path + '/positive/',
                                                            newly_labled_path + '/negative/')
                emb_dataset = random.sample(emb_dataset, len(emb_dataset))
                n_80 = (len(emb_dataset) * 8) // 10
                training_dataset = DataLoader(emb_dataset[:n_80], batch_size=32)  # TODO yml
                validation_dataset = DataLoader(emb_dataset[n_80 + 1:], batch_size=1)
                train_models.train_linear(training_dataset, validation_dataset)
                # {self.model_path}AL_0
                # strategy_embeddings, strategy_images = activelabeler.get_images_to_label_offline(
                #     train_models.get_model(), "uncertainty", parameters['ActiveLabeler']['sample_size'], None, "cuda")

            #put seed dataset/newly labeled data in archive path and clear newly labeled
            for img in list(paths.list_images(newly_labled_path + "/positive")):
                shutil.copy(img, parameters['AL_main']['archive_path'] + "/positive")
            for img in list(paths.list_images(newly_labled_path + "/negative")):
                shutil.copy(img, parameters['AL_main']['archive_path'] + "/negative")
            newly_labled_path = parameters['AL_main']['newly_labled_path']
            for img in list(paths.list_images(newly_labled_path)):
                os.remove(img)

            if input_counter == 'f':
                model_type = "encoder"
                #put seed in archive path
                # train all = create dataloader on archive dataset & split , train all  , get uncertain
                model_type = "encoder"
                archive_dataset = torchvision.datasets.ImageFolder(parameters['AL_main']['archive_path'], t)
                n_80 = (len(archive_dataset) * 8) // 10
                n_20 = len(archive_dataset) - n_80
                training_dataset, validation_dataset = torch.utils.data.random_split(archive_dataset, [n_80, n_20])
                training_dataset = DataLoader(training_dataset, batch_size=32)
                validation_dataset = DataLoader(validation_dataset, batch_size=1)
                train_models.train_all(training_dataset, validation_dataset)
                #change generate emb again => using encoder from model from train_all
                # emb from unlabeled pool => need to update AL with new emb ??
                # x = torchvision.datasets.ImageFolder(parameters['AL_main']['archive_path'], t)
                # encoder = train_models.get_model().encoder(x)[-1]
                logging.info('initialize_embeddings')
                self.initialize_emb_counter += 1
                parameters['annoy']['annoy_path'] = parameters['annoy'][
                                                        'annoy_path'] + f"{self.initialize_emb_counter}"
                encoder =  train_models.get_model()
                self.initialize_embeddings(parameters['model']['image_size'], parameters['model']['embedding_size'],
                                           encoder,
                                           [ parameters['data']['data_path'] + "/Unlabeled/" + image_name for image_name in self.unlabeled_list ],
                                           parameters['annoy']['num_nodes'],
                                           parameters['annoy']['num_trees'], parameters['annoy']['annoy_path'],"encoder")
                #update AL class with new emb
                activelabeler.get_embeddings_offline(self.create_emb_mapping(self.unlabeled_list),
                                                 [parameters['data']['data_path'] + "/Unlabeled/" + image_name for
                                                  image_name in self.unlabeled_list])

            # AL.getimgstolabel => uncertain imgs => nn
            curr_model = model if model_type =="model" else encoder
            strategy_embeddings, strategy_images = activelabeler.get_images_to_label_offline(
                curr_model, "uncertainty", parameters['ActiveLabeler']['sample_size'], None, "cuda")

            imgs = self.search_similar(strategy_images, int(parameters['AL_main']['n_closest']),
                                       parameters['annoy']['num_nodes'], parameters['annoy']['annoy_path'],
                                       None, curr_model,model_type) #TODO inference model this works
            # tmp1 = set(strategy_images)
            tmp2 = set(imgs)
            tmp2.update(strategy_images)
            imgs = list(tmp2)

            self.label_data(imgs, parameters['data']['data_path'], parameters['nn']['swipe_url'],
                            parameters['nn']['simulate_label'],
                            parameters['nn']['unlabled_path'], parameters['AL_main']['newly_labled_path'],
                            parameters['AL_main']['newly_labled_path'] + "/positive",
                            parameters['AL_main']['newly_labled_path'] + "/negative", None)

            #update embeddings with unlabeled image embeddings #TODO check initializations
            activelabeler.get_embeddings_offline(self.create_emb_mapping(self.unlabeled_list), [ parameters['data']['data_path'] + "/Unlabeled/" + image_name for image_name in self.unlabeled_list ])


        # iteration= 0
        # newly_labled_path = parameters['nn']['labeled_path']
        # while True:
        #     iteration +=1
        #     print(f"iteration {iteration}")
        #     # entire model - image_dataloader
        #     # offline, train_linear
        #
        #     #label - emb
        #     # lists emb , labels - 0(neg),1(pos) , labled image names - not req AL => mapped via indices
        #     #emb_dataset = [[emb,label]..]
        #     #newly_labeld
        #     #sample 80:20
        #     emb_dataset = self.create_emb_label_mapping(newly_labled_path + '/positive/',newly_labled_path + '/negative/')
        #     emb_dataset = random.sample(emb_dataset, len(emb_dataset))
        #     n_80 = (len(emb_dataset)*8)//10
        #     training_dataset = DataLoader(emb_dataset[:n_80], batch_size = 32) #TODO yml
        #     validation_dataset = DataLoader(emb_dataset[n_80+1:], batch_size = 1)
        #     train_models.train_linear(training_dataset, validation_dataset)
        #     #{self.model_path}AL_0
        #     strategy_embeddings, strategy_images= activelabeler.get_images_to_label_offline(train_models.get_model(), "uncertainty", parameters['ActiveLabeler']['sample_size'], None, "cuda")
        #
        #     #nn for each emb
        #     # label
        #     #archive => pos , neg
        #     imgs = self.search_similar(strategy_images, int(parameters['AL_main']['n_closest']),parameters['annoy']['num_nodes'],parameters['annoy']['annoy_path'], parameters['data']['data_path'],model)
        #     #tmp1 = set(strategy_images)
        #     tmp2 = set(imgs)
        #     tmp2.update(strategy_images)
        #     imgs = list(tmp2)
        #
        #     for img in list(paths.list_images(newly_labled_path+ "/positive")):
        #         shutil.copy(img, parameters['AL_main']['archive_path']+ "/positive")
        #     for img in list(paths.list_images(newly_labled_path + "/negative")):
        #         shutil.copy(img, parameters['AL_main']['archive_path']+ "/negative")
        #     newly_labled_path = parameters['AL_main']['newly_labled_path']
        #     for img in list(paths.list_images(newly_labled_path)):
        #         os.remove(img)
        #
        #     self.label_data(imgs, parameters['data']['data_path'], parameters['nn']['swipe_url'], parameters['nn']['simulate_label'],
        #                     parameters['nn']['unlabled_path'], parameters['AL_main']['newly_labled_path'], parameters['AL_main']['newly_labled_path']+"/positive",
        #                     parameters['AL_main']['newly_labled_path']+"/negative", None)
        #
        #
        #     #reset embeddings with unlabeled images
        #     activelabeler.get_embeddings_offline(self.create_emb_mapping(self.unlabeled_list), [ parameters['data']['data_path'] + "/Unlabeled/" + image_name for image_name in self.unlabeled_list ])
        #
        #     print("Enter c to continue") #TODO unrecognizable char
        #     input_counter = input()
        #     if input_counter != 'c': break
        #
        # for img in list(paths.list_images(newly_labled_path+ "/positive")):
        #     shutil.copy(img, parameters['AL_main']['archive_path']+ "/positive")
        # for img in list(paths.list_images(newly_labled_path + "/negative")):
        #     shutil.copy(img, parameters['AL_main']['archive_path']+ "/negative")
        # newly_labled_path = parameters['AL_main']['newly_labled_path']
        # for img in list(paths.list_images(newly_labled_path)):
        #     os.remove(img)
        #
        # #only for offline
        # #TODO model_confident
        # if iteration>=3:
        #     #  archive
        #     # sample 80:20 #TODO
        #     logging.info("model_confident")
        #     archive_dataset = torchvision.datasets.ImageFolder(parameters['AL_main']['archive_path'], t)
        #     n_80 = (len(archive_dataset) * 8)//10
        #     n_20 = len(archive_dataset) - n_80
        #     training_dataset, validation_dataset= torch.utils.data.random_split(archive_dataset,[n_80,n_20])
        #     training_dataset = DataLoader(training_dataset, batch_size = 32)
        #     validation_dataset = DataLoader(validation_dataset, batch_size = 1)
        #     train_models.train_all(training_dataset, validation_dataset)
        #     #emb from unlabeled pool, AL.getimgstolabel => uncertain imgs => nn   => ask user linear or entire ,
        #     #self.initialize_embeddings(image_size, embedding_size, train_models.get_model(), dataset_paths, num_nodes, num_trees, annoy_path)
        #     #forward pass - get_images_to_label_offline()
        #     #label
        #     #linear layer
        #

