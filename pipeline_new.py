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
from sklearn.metrics import precision_score,recall_score, f1_score, accuracy_score
import time
import pandas as pd

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
from tsne import  TSNE_visualiser

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
    def __init__(self, config_path,class_name):
        self.config_path = config_path
        self.dataset_paths=[]
        self.unlabeled_list = []
        self.labled_list = []
        self.embeddings = None
        self.div_embeddings = None
        self.initialize_emb_counter =0
        self.class_name = class_name
        # self.metrics = {"class": [],"step": [],"model_type": [], "f1_score":[], "precision":[],"accuracy":[], "recall":[],"train_ratio":[],"pos_train_img":[],"neg_train_imgs":[], "train_time":[],
        #                 "actual_pos_imgs":[],"pos_class_confidence_0.8":[],"pos_class_confidence_0.5":[],"pos_class_confidence_median":[],"actual_neg_imgs":[],"neg_class_confidence_0.8":[],"neg_class_confidence_0.5":[],"neg_class_confidence_median":[] }
        # self.metrics = {"class": [], "step": [], "model_type": [], "f1_score": [], "precision": [], "accuracy": [],
        #                 "recall": [], "train_ratio": [], "pos_train_img": [], "neg_train_imgs": [], "train_time": [],
        #                 "pos_class_confidence_0.8": [], "pos_class_confidence_0.5": [],
        #                 "pos_class_confidence_median": [], "neg_class_confidence_0.8": [],
        #                 "neg_class_confidence_0.5": [], "neg_class_confidence_median": [],
        #                 "class_confidence_0.8": [], "class_confidence_0.5": [],
        #                 "class_confidence_median": [], "actual_pos_imgs_0.8": [], "actual_pos_imgs_0.5": []}
        self.metrics = {"class": [], "step": [], "model_type": [],"train_ratio": [], "pos_train_img": [], "neg_train_imgs": [], "train_time": [],"labled_pos": [],"labled_neg": [], "f1_score": [], "precision": [], "accuracy": [],
                        "recall": [], "pos_class_confidence_0.8": [], "pos_class_confidence_0.5": [],
                        "pos_class_confidence_median": [], "neg_class_confidence_0.8": [],
                        "neg_class_confidence_0.5": [], "neg_class_confidence_median": [], "total_labeling_effort": [],
                        "actual_pos_imgs_0.8": [], "actual_pos_imgs_0.5": []} #, "AL_score":[]}
        self.prediction_prob ={}

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
        x = model(im) if model_type == "model" else model.encoder(im)[-1]
        return x[0]


    def get_nn_annoy(self, image_path, n_closest, num_nodes, annoy_path, data_path,model, model_type,disp=False):
        # load dependencies
        u = AnnoyIndex(num_nodes, 'euclidean')
        #print(annoy_path)
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
                   negative_path=None, unsure_path=None,class_name="airplane"):

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
        #TODO swipe labeler
        self.swipe_label_simulate(swipe_url=swipe_url,simulate_label=simulate_label,unlabled_path=unlabeled_path, labeled_path=labeled_path, positive_path=postive_path,
                                  negative_path=negative_path, unsure_path=unsure_path,class_name=class_name)

    def swipe_label_simulate(self, swipe_url, simulate_label=False,unlabled_path=None, labeled_path=None, positive_path=None, negative_path=None,
                             unsure_path=None,class_name="airplane"):
        logging.info("Calling swipe labeler")
        print(f'\n {len(list(paths.list_images(unlabled_path)))} images to label. Go to {swipe_url}')

        ori_labled = len(list(paths.list_images(labeled_path)))
        ori_pos = len(list(paths.list_images(positive_path)))
        ori_neg = len(list(paths.list_images(negative_path)))

        if simulate_label: # TODO
            for img in list(paths.list_images(unlabled_path)):
                src = unlabled_path + "/" + img.split('/')[-1]
                dest = (positive_path + "/" + img.split('/')[-1]) if self.class_name in img else (
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
                             unsure_path=None,class_name="airplane"):
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
            r_imgs = random.sample(tmp3, k=n_20)
            imgs = imgs + r_imgs

            self.label_data(imgs, data_path,swipe_url,simulate_label, unlabled_path, labeled_path, positive_path,
                                  negative_path, unsure_path,class_name)

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
        if positive_path is None:
            pos_label = []
        else:
            pos_label = [i.split("/")[-1] for i in list(paths.list_images(positive_path))]
        if negative_path is None:
            neg_label = []
        else:
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

    def test_data(self, model,test_path,t,device="cuda"):
        test_dataset = torchvision.datasets.ImageFolder(test_path, t)
        loader = DataLoader(test_dataset, batch_size=1)
        #TODO confirm metrics with anirudh

        #model.to(device)
        model.eval()
        op = []
        gt = []
        with torch.no_grad():
            for step, (x, y) in enumerate(loader):
                x = x.to(device)
                y = y.to(device)
                feats = model.encoder(x)[-1]
                feats = feats.view(feats.size(0), -1)
                output = model.linear_model(feats)
                inds = torch.argmax(output, dim=1)
                # print(output[0:5])
                # print(inds[0:5])
                #op.append(inds.item())
                op.append(output.detach().cpu().numpy())
                gt.append(y.item())
            pred = []
            for i in op:
                if i[0] <= 0.5:
                    pred.append(0)
                else:
                    pred.append(1)
            op = pred
            # print(op)
            prec = precision_score(gt, op)
            rec = recall_score(gt, op)
            f1 = f1_score(gt, op)
            acc = accuracy_score(gt,op)
            # write code to append to dictionary
            #         self.metrics = {"class": [],"step": [],"model_type": [], "f1_score":[], "precision":[],"recall":[] }
            #step, class append, model_type in main
            self.metrics["f1_score"].append(f1)
            self.metrics["precision"].append(prec)
            self.metrics["recall"].append(rec)
            self.metrics["accuracy"].append(acc)



    @property
    def main(self):
        # offline
        # TODO printing and logging

        # +++++++++++++++++++++++++++++++++++++++++++++++++++++
        #seed dataset

        logging.info('load config')
        parameters = self.load_config(self.config_path)

        logging.info('load model')
        model = self.load_model(parameters['model']['model_type'], parameters['model']['model_path'], parameters['data']['data_path'])

        logging.info('initialize_embeddings')
        self.initialize_emb_counter +=1 #TODO use _0 and then iteration number for later
        parameters['annoy']['annoy_path'] = parameters['annoy']['annoy_path'] + f"{self.initialize_emb_counter}" + ".ann"
        self.initialize_embeddings(parameters['model']['image_size'], parameters['model']['embedding_size'], model,
                                   list(paths.list_images(parameters['data']['data_path'])), parameters['annoy']['num_nodes'],
                                   parameters['annoy']['num_trees'], parameters['annoy']['annoy_path'])

        self.dataset_paths = list(paths.list_images(parameters['data']['data_path']))
        self.unlabeled_list = [i.split('/')[-1] for i in self.dataset_paths]
        if  parameters['Continuation']['nn']==1:
            logging.info('create_seed_dataset')
            self.labled_list = []
            self.create_seed_dataset(parameters['nn']['ref_img_path'],parameters['data']['data_path'],parameters['nn']['swipe_url'],parameters['nn']['simulate_label'],parameters['annoy']['num_nodes'],parameters['annoy']['annoy_path'] ,model,
                                     parameters['nn']['unlabled_path'],parameters['nn']['labeled_path'],parameters['nn']['positive_path'],parameters['nn']['negative_path'],parameters['nn']['unsure_path'],self.class_name)
            newly_labled_path = parameters['nn']['labeled_path']

        else:
            self.labled_list = [i.split('/')[-1] for i in list(paths.list_images(parameters['Continuation']['seed_data_path']))]
            for i in self.labled_list:
                self.unlabeled_list.remove(i)
            newly_labled_path =  parameters['Continuation']['seed_data_path']

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
        #newly_labled_path = parameters['nn']['labeled_path']  #seed dataset
        while True:
            iteration += 1
            print(f"iteration {iteration}")

            print("Enter l for Linear, f for finetuning and q to quit")  # TODO unrecognizable char

            #input_counter = input()

            # automatic iteration
            input_counter = 'f'
            if iteration == 21: input_counter = 'q'

            if input_counter == 'q': break

            if input_counter == 'l':
                #linear = create newly labeled emb, sample and split , get uncertain
                emb_dataset = self.create_emb_label_mapping(newly_labled_path + '/positive/',
                                                            newly_labled_path + '/negative/')

                # if iteration==1:
                #     emb_dataset_archive = self.create_emb_label_mapping(
                #         parameters['nn']['labeled_path'] + "/positive",
                #         None)
                #     emb_dataset_archive_r_p = random.sample(emb_dataset_archive,
                #                                           min(len(emb_dataset) // 9, len(emb_dataset_archive)))
                #     emb_dataset_archive = self.create_emb_label_mapping(
                #         None,
                #         parameters['nn']['labeled_path'] + "/negative")
                #     emb_dataset_archive_r_n = random.sample(emb_dataset_archive,
                #                                           min(len(emb_dataset) // 9, len(emb_dataset_archive)))
                #
                # else:
                #     emb_dataset_archive = self.create_emb_label_mapping(parameters['AL_main']['archive_path'] + "/positive",None)
                #     emb_dataset_archive_r_p =  random.sample(emb_dataset_archive, min(len(emb_dataset)//9,len(emb_dataset_archive)))
                #     emb_dataset_archive = self.create_emb_label_mapping(
                #         None,
                #         parameters['AL_main']['archive_path'] + "/negative")
                #     emb_dataset_archive_r_n = random.sample(emb_dataset_archive,
                #                                           min(len(emb_dataset) // 9, len(emb_dataset_archive)))
                #
                # for i in emb_dataset_archive_r_n:
                #     emb_dataset.append(i)
                # for i in emb_dataset_archive_r_p:
                #     emb_dataset.append(i)

                if iteration==1:
                    emb_dataset_archive = self.create_emb_label_mapping(
                        parameters['nn']['labeled_path'] + "/positive",
                        parameters['nn']['labeled_path'] + "/negative")

                else:
                    emb_dataset_archive = self.create_emb_label_mapping(parameters['AL_main']['archive_path'] + "/positive", parameters['AL_main']['archive_path'] + "/negative")

                for i in emb_dataset_archive:
                    emb_dataset.append(i)

                #"train_ratio": [], "pos_train_img": [], "neg_train_imgs": []}
                tmp_p = len(list(paths.list_images(newly_labled_path + "/positive"))) + len(list(paths.list_images(parameters['AL_main']['archive_path'] + "/positive")))
                tmp_n = len(list(paths.list_images(newly_labled_path + "/negative"))) + len(list(paths.list_images(parameters['AL_main']['archive_path'] + "/negative")))
                self.metrics["pos_train_img"].append(tmp_p)
                self.metrics["neg_train_imgs"].append(tmp_n)
                tmp = tmp_n / tmp_p if tmp_p > 0 else 0
                self.metrics["train_ratio"].append(tmp)

                emb_dataset = random.sample(emb_dataset, len(emb_dataset))
                n_80 = (len(emb_dataset) * 8) // 10
                training_dataset = DataLoader(emb_dataset[:n_80], batch_size=32)  # TODO yml
                validation_dataset = DataLoader(emb_dataset[n_80 + 1:], batch_size=1)
                tic = time.perf_counter()
                train_models.train_linear(training_dataset, validation_dataset)
                toc = time.perf_counter()
                self.metrics["train_time"].append((toc - tic)//60)
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

                # "train_ratio": [], "pos_train_img": [], "neg_train_imgs": []}
                tmp_p = len(list(paths.list_images(parameters['AL_main']['archive_path'] + "/positive"))) +  len(list(paths.list_images(newly_labled_path + "/positive")))
                tmp_n = len(list(paths.list_images(parameters['AL_main']['archive_path'] + "/negative"))) + len(list(paths.list_images(newly_labled_path + "/negative")))
                self.metrics["pos_train_img"].append(tmp_p)
                self.metrics["neg_train_imgs"].append(tmp_n)
                tmp = tmp_n / tmp_p if tmp_p > 0 else 0
                self.metrics["train_ratio"].append(tmp)

                archive_dataset = torchvision.datasets.ImageFolder(parameters['AL_main']['archive_path'], t)
                n_80 = (len(archive_dataset) * 8) // 10
                n_20 = len(archive_dataset) - n_80
                training_dataset, validation_dataset = torch.utils.data.random_split(archive_dataset, [n_80, n_20])
                training_dataset = DataLoader(training_dataset, batch_size=32)
                validation_dataset = DataLoader(validation_dataset, batch_size=1)
                tic = time.perf_counter()
                train_models.train_all(training_dataset, validation_dataset)
                toc = time.perf_counter()
                self.metrics["train_time"].append((toc - tic)//60)
                #change generate emb again => using encoder from model from train_all
                # emb from unlabeled pool => need to update AL with new emb ??
                # x = torchvision.datasets.ImageFolder(parameters['AL_main']['archive_path'], t)
                # encoder = train_models.get_model().encoder(x)[-1]
                logging.info('initialize_embeddings')
                self.initialize_emb_counter += 1
                parameters['annoy']['annoy_path'] = parameters['annoy'][
                                                        'annoy_path'] + f"{self.initialize_emb_counter}" + ".ann"
                encoder =  train_models.get_model().to("cuda") #TODO device
                self.dataset_paths = [ parameters['data']['data_path'] + "/Unlabeled/" + image_name for image_name in self.unlabeled_list ]
                self.initialize_embeddings(parameters['model']['image_size'], parameters['model']['embedding_size'],
                                           encoder,
                                           self.dataset_paths,
                                           parameters['annoy']['num_nodes'],
                                           parameters['annoy']['num_trees'], parameters['annoy']['annoy_path'],"encoder")

                # #update AL class with new emb
                mapping = self.create_emb_mapping(self.unlabeled_list)
                activelabeler.get_embeddings_offline(mapping,
                                                     [parameters['data']['data_path'] + "/Unlabeled/" + image_name for
                                                      image_name in self.unlabeled_list])

                # mapping =[]
                # for i in range(len(self.unlabeled_list)):
                #     mapping.append(self.embeddings[i])
                # activelabeler.get_embeddings_offline(mapping,
                #                                  [parameters['data']['data_path'] + "/Unlabeled/" + image_name for
                #                                   image_name in self.unlabeled_list])

            # AL.getimgstolabel => uncertain imgs => nn sampling_strategy
            curr_model = model if model_type =="model" else encoder

            #unlabled images redundant
            if os.path.exists(parameters["test"]["evaluation_path"]):
                shutil.rmtree(parameters["test"]["evaluation_path"])
            pathlib.Path(parameters["test"]["evaluation_path"]+ "/combined/Unlabeled").mkdir(parents=True, exist_ok=True)

            for img in self.unlabeled_list:
                src = "/content/Dataset/Unlabeled" + "/" + img.split('/')[-1]
                dest = ("/content/Evaluation_Data/combined/Unlabeled"  + "/" + img.split('/')[-1])
                shutil.copy(src, dest)

            strategy_embeddings, strategy_images = activelabeler.get_images_to_label_offline(
                train_models.get_model(),parameters['ActiveLabeler']['sampling_strategy'] , parameters['ActiveLabeler']['sample_size'], None, "cuda",parameters["test"]["evaluation_path"]+"/combined")

            #train_models.get_model().train_encoder=False
            # train_models.get_model().freeze_encoder()
            #print(strategy_images)
            if parameters['AL_main']['nn']==1:

                imgs = self.search_similar(strategy_images, int(parameters['AL_main']['n_closest']),
                                           parameters['annoy']['num_nodes'], parameters['annoy']['annoy_path'],
                                           None, curr_model,model_type) #TODO inference model this works
                print("nn imgs ", imgs)
                # tmp1 = set(strategy_images)
                tmp2 = set(imgs)
                print("len nn", len(tmp2))
                tmp2.update(strategy_images)
                imgs = list(tmp2)
                print("len nn + strategy imgs", len(tmp2))

            else:
                imgs = strategy_images

            self.label_data(imgs, parameters['data']['data_path'], parameters['nn']['swipe_url'],
                            parameters['nn']['simulate_label'],
                            parameters['nn']['unlabled_path'], parameters['AL_main']['newly_labled_path'],
                            parameters['AL_main']['newly_labled_path'] + "/positive",
                            parameters['AL_main']['newly_labled_path'] + "/negative", None,self.class_name)

            tmp1 = len(list(paths.list_images(parameters['AL_main']['archive_path'] + "/positive")))
            tmp2 = len(list(paths.list_images(newly_labled_path + "/positive")))
            tmp3 = len(list(paths.list_images(parameters['AL_main']['archive_path'] + "/negative")))
            tmp4 = len(list(paths.list_images(newly_labled_path + "/negative")))
            print(f"Total Images: {tmp1} + {tmp2} = {tmp1+tmp2} positive || {tmp3} + {tmp4} = {tmp3+tmp4} negative")

            self.metrics['labled_pos'].append(tmp2)
            self.metrics['labled_neg'].append(tmp4)

            #update embeddings with unlabeled image embeddings #TODO check initializations
            #mapping = []
            # if model_type=="encoder": #TODO model type doesnt change after every iteration, even after (...=> f => linear) will this work ?
            #     for i in range(len(self.unlabeled_list)):
            #         mapping.append(self.embeddings[i])
            # else:
            mapping = self.create_emb_mapping(self.unlabeled_list)
            activelabeler.get_embeddings_offline(mapping, [ parameters['data']['data_path'] + "/Unlabeled/" + image_name for image_name in self.unlabeled_list ])

            #--TEST
            # step, class, model_type append in main
            self.metrics["step"].append(iteration)
            self.metrics["class"].append(self.class_name)
            self.metrics["model_type"].append(input_counter)
            self.test_data(train_models.get_model(),parameters["test"]["test_path"],t)

            #TODO redundant
            if os.path.exists(parameters["test"]["evaluation_path"]):
                shutil.rmtree(parameters["test"]["evaluation_path"])
            pathlib.Path(parameters["test"]["evaluation_path"]+ "/positive/Unlabeled").mkdir(parents=True, exist_ok=True)
            pathlib.Path(parameters["test"]["evaluation_path"] + "/negative/Unlabeled").mkdir(parents=True, exist_ok=True)

            for img in self.unlabeled_list:
                src = "/content/Dataset/Unlabeled" + "/" + img.split('/')[-1]
                dest = ("/content/Evaluation_Data/positive/Unlabeled"  + "/" + img.split('/')[-1]) if self.class_name in img else (
                        "/content/Evaluation_Data/negative/Unlabeled"  + "/" + img.split('/')[-1])
                shutil.copy(src, dest)

            #TSNE #TODO os.path.join
            # embeddings = self.create_emb_mapping(self.labled_list)
            # ims = [parameters["data"]["data_path"] + "/Unlabeled/" + i for i.split('/')[-1] in self.labled_list]
            # tsne = TSNE_visualiser(embeddings, ims)
            # result = tsne.fit_tsne()
            # tsne.tsne_to_grid_plotter_manual(result[:, 0], result[:, 1], tsne.filenames)


            tmp_prob= activelabeler.get_probablities(parameters["test"]["evaluation_path"]+"/positive",train_models.get_model(),0.8,parameters['model']['image_size'])
            print("tmp_prob- probs - sorted-pos", sorted(tmp_prob))
            tmp_c = tmp_prob
            count_8 = 0
            count_5 = 0
            tmp_prob2 = []
            for i in range(len(tmp_prob)):
                tmp_prob2.append(tmp_prob[i][0])
                if tmp_prob[i][0] >= 0.8:
                    count_8 += 1
                if tmp_prob[i][0] >= 0.5:
                    count_5 += 1
            self.metrics["pos_class_confidence_0.8"].append(count_8)
            self.metrics["pos_class_confidence_0.5"].append(count_5)
            self.metrics["pos_class_confidence_median"].append(np.median(tmp_prob2))
            #will be unlabled pos imgs
            #self.metrics["actual_pos_imgs"].append(len(list(paths.list_images(parameters["test"]["evaluation_path"]+"/positive"))))

            tmp_prob = activelabeler.get_probablities(parameters["test"]["evaluation_path"] + "/negative",
                                                      train_models.get_model(), 0.8, parameters['model']['image_size'])
            print("tmp_prob- probs - sorted-neg", sorted(tmp_prob))
            tmp_c = tmp_c + tmp_prob
            print("tmp_c- probs - sorted-combined", sorted(tmp_c))
            count_8 = 0
            count_5 = 0
            tmp_prob3 = []
            for i in range(len(tmp_prob)):
                tmp_prob3.append(tmp_prob[i][0])
                if tmp_prob[i][0] >= 0.8:
                    count_8 += 1
                if tmp_prob[i][0] >= 0.5:
                    count_5 += 1
            self.metrics["neg_class_confidence_0.8"].append(count_8)
            self.metrics["neg_class_confidence_0.5"].append(count_5)
            self.metrics["neg_class_confidence_median"].append(np.median(tmp_prob3))
            #self.metrics["actual_neg_imgs"].append(len(list(paths.list_images(parameters["test"]["evaluation_path"] + "/negative"))))

            self.metrics["actual_pos_imgs_0.8"].append(self.metrics["pos_train_img"][-1]+self.metrics["pos_class_confidence_0.8"][-1]+self.metrics['labled_pos'][-1])
            self.metrics["actual_pos_imgs_0.5"].append(self.metrics["pos_train_img"][-1]+self.metrics["pos_class_confidence_0.5"][-1]+self.metrics['labled_pos'][-1])
            self.metrics["total_labeling_effort"].append(self.metrics["pos_train_img"][-1]+self.metrics["neg_train_imgs"][-1]+self.metrics["pos_class_confidence_0.8"][-1]+self.metrics['neg_class_confidence_0.8'][-1])
            #self.metrics["AL_score"].append(self.metrics["actual_pos_imgs_0.8"][-1]/self.metrics["total_labeling_effort"][-1])

            tmp_prob2.extend(tmp_prob3)
            #TODO add config path
            #tmp_df = pd.DataFrame(tmp_prob2, columns = [f'{iteration}'])
            self.prediction_prob[iteration]=tmp_prob2
            # print("prediciton_prob",self.prediction_prob)
            df = pd.DataFrame.from_dict(self.prediction_prob, orient='index').transpose()
            df.to_csv(parameters["test"]["prob_csv_path"],index=False)

            # #--- forward pass on whole dataset
            # imgs, tmp_prob = activelabeler.get_probablities(parameters["data"]["data_path"],
            #                                           train_models.get_model(), 0.8, parameters['model']['image_size'],paths=True)
            # # print("final prob", tmp_prob)
            # count_8 = 0
            # count_5 = 0
            # tmp_prob2 = []
            # tmp_pos, tmp_pos2 = 0, 0
            # #imgs = list(paths.list_images(parameters["data"]["data_path"]))
            # for i in range(len(tmp_prob)):
            #     tmp_prob2.append(tmp_prob[i][0])
            #     if tmp_prob[i][0] >= 0.8:
            #         count_8 += 1
            #         if (self.class_name in imgs[i]):
            #             tmp_pos += 1
            #     if tmp_prob[i][0] >= 0.5:
            #         count_5 += 1
            #         if (self.class_name in imgs[i]):
            #             tmp_pos2 += 1
            #
            # self.metrics["class_confidence_0.8"].append(count_8)
            # self.metrics["class_confidence_0.5"].append(count_5)
            # self.metrics["class_confidence_median"].append(np.median(tmp_prob2))
            # tmp, tmp2 = 0, 0
            # # actual positive imgs = imgs that are actually pos out of the imgs model predicted as positive
            # self.metrics["actual_pos_imgs_0.8"].append(tmp_pos)
            # self.metrics["actual_pos_imgs_0.5"].append(tmp_pos2)


            print(f"iteration {iteration} metrics = {self.metrics}")
            df = pd.DataFrame.from_dict(self.metrics, orient='index').transpose()
            # rounding to 2
            col_names = ['f1_score', 'precision', 'accuracy', 'recall', 'train_ratio', 'pos_class_confidence_median',
                         'neg_class_confidence_median',]#'AL_score']
            for i in col_names:
                df[i] = df[i].astype(float).round(2)
            df.to_csv(parameters["test"]["metric_csv_path"],index=False)



        #---final forward pass on whole dataset
        # tmp_prob = activelabeler.get_probablities(parameters["data"]["data_path"],
        #                                           train_models.get_model(), 0.8, parameters['model']['image_size'])
        # print("final prob",tmp_prob)
        # count_8 = 0
        # count_5 = 0
        # tmp_prob2 = []
        # tmp_pos,tmp_pos2 = 0,0
        # imgs = list(paths.list_images(parameters["data"]["data_path"]))
        # for i in range(len(tmp_prob)):
        #     tmp_prob2.append(tmp_prob[i][0])
        #     if tmp_prob[i][0] >= 0.8:
        #         count_8 += 1
        #         if(self.class_name in imgs[i]):
        #             tmp_pos +=1
        #     if tmp_prob[i][0] >= 0.5:
        #         count_5 += 1
        #         if (self.class_name in imgs[i]):
        #             tmp_pos2 += 1
        # tmp_metrics ={  "class_confidence_0.8": [], "class_confidence_0.5": [],
        #                 "class_confidence_median": [], "actual_pos_imgs_0.8": [],"actual_pos_imgs_0.5": []}
        #
        # tmp_metrics["class_confidence_0.8"].append(count_8)
        # tmp_metrics["class_confidence_0.5"].append(count_5)
        # tmp_metrics["class_confidence_median"].append(np.median(tmp_prob2))
        # tmp,tmp2 = 0,0
        # #actual positive imgs = imgs that are actually pos out of the imgs model predicted as positive
        # tmp_metrics["actual_pos_imgs_0.8"].append(tmp_pos)
        # tmp_metrics["actual_pos_imgs_0.5"].append(tmp_pos2)
        #
        # print(f"iteration {iteration} final metrics = {tmp_metrics}")
        # df = pd.DataFrame.from_dict(tmp_metrics, orient='index').transpose()
        # df.to_csv(parameters["test"]["metric_csv_path"],mode='a')
        # print("done")

        #todo final newlylabled to archive folder
        #--CSV = metrics to csv conversion
        # df = pd.DataFrame.from_dict(self.metrics, orient='index').transpose()
        # #rounding to 2
        # col_names = ['f1_score', 'precision', 'accuracy', 'recall', 'train_ratio', 'pos_class_confidence_median',
        #              'neg_class_confidence_median']
        # for i in col_names:
        #     df[i] = df[i].astype(float).round(2)
        # df.to_csv(parameters["test"]["metric_csv_path"], mode='a', header=not os.path.exists(parameters["test"]["metric_csv_path"]))

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

