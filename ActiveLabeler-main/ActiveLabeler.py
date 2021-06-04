import os
import random
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder


random.seed(100)


class ActiveLabeler():
    def __init__(self, embeddings, images_path):
        self.embeddings = embeddings
        self.images_path = images_path
    
    def strategy(self, name, query, N, strategy_params):
        """Method used to get a subset of unlabled images.

        :param name: Name of the sampling strategy to use
        :options name: uncertainty, random, positive

        :param query: A num_images X num_classes sized numpy array. This numpy array has the softmax probability 
                    of each image. These softmax probs are predicted by the latest finetune model.
        :type query: numpy array
        
        :param N: Number of images that should be in the subset.
        :type N: integer

        :returns list/numpy array of the index of images in the sebset. 
        """
        if name == 'uncertainty':
            return np.argsort(query.std(axis = 1))[:N]
        elif name == 'random':
            return [random.randrange(0, len(query), 1) for i in range(N)]
        elif name == 'positive':
            positive_predictions = np.array([query[i][strategy_params["class_label"]] for i in range(len(query))])
            positive_index = np.argsort(positive_predictions)
            return positive_index[::-1][:N]
        else:
            raise NotImplementedError

    def get_embeddings_offline(self, emb, data_paths):
        self.embeddings = emb
        self.images_path = data_paths

    def get_images_to_label_offline(self, model, sampling_strat, sample_size, strategy_params, device):
        #Load stuff
        model.eval()
        if device == "cuda":
            model.cuda()
        dataset = self.embeddings
        image_paths = self.images_path

        #Forward Pass
        with torch.no_grad():
            bs = 128
            if len(dataset) < bs:
                bs = 1
            loader = DataLoader(dataset, batch_size=bs, shuffle=False)
            model_predictions = []
            for batch in tqdm(loader):
                if device == "cuda":
                    x = torch.FloatTensor(batch).to(device)  # torch.cuda.FloatTensor(batch)
                else:
                    x = torch.FloatTensor(batch)
                predictions = model.linear_model(x)
                model_predictions.extend(predictions.cpu().detach().numpy())

        #Strategy
        model_predictions = np.array(model_predictions)
        subset = self.strategy(sampling_strat, model_predictions, sample_size, strategy_params)

        #Stuff to return
        strategy_embeddings = np.array([i for i in dataset])[subset]
        strategy_images = np.array([i for i in image_paths])[subset]

        return strategy_embeddings, strategy_images

    def get_probablities(self, DATA_PATH, model, prob,image_size):

        #model = CLASSIFIER.CLASSIFIER.load_from_checkpoint(MODEL_PATH)
        #pass as data_path
        #evaluation data_p / pos => all positive images => how many imgs model thinks is pos with accuracy of 0.8 => pos_class_confidence
        #evaluation data_n /neg  =>  all neg images => same => neg_class_confidence_0.8
        #pos,neg => 0.8 0.5  = prob

        unlabled_probablites = []
        model.eval()
        model.cuda()

        def to_tensor(pil):
            return torch.tensor(np.array(pil)).permute(2, 0, 1).float()

        t = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.Lambda(to_tensor)
        ])
        dataset = ImageFolder(DATA_PATH, transform=t)
        with torch.no_grad():
            bs = 128
            if len(dataset) < bs:
                bs = 1
            loader = DataLoader(dataset, batch_size=bs, shuffle=False)
            for batch in tqdm(loader):
                x = batch[0].cuda()
                predictions = model(x)
                unlabled_probablites.extend(predictions.detach().cpu().numpy())
        positive_predictions = np.array([unlabled_probablites[i][1] for i in range(len(unlabled_probablites))])
        count = 0
        for i in positive_predictions:
            if i > prob:
                count += 1 #TODO count , median
        #median positive_predictions np.median(pos_predic)
        return positive_predictions