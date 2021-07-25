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
        if name == 'gaussian':
            def gaussian(x, mean, sigma):
                a = (1 / (sigma * np.sqrt(2 * np.pi)))
                exp_val = - (1 / 2) * (((x - mean) ** 2) / (sigma ** 2))

                return (a * np.exp(exp_val))

            def get_gaussian_weights(confidences):
                mean = 0.5  # TUNABLE: This is where you want your peak.
                sigma = 0.2  # TUNABLE: This defines how exponential your gaussian curve is around the peak.
                return [gaussian(conf_i, mean, sigma) for conf_i in confidences]

            def get_samples(image_list, confidences,number_of_samples):
                sample_wts = get_gaussian_weights(confidences)  # N weight values corresponding to each image
                picked_samples = np.random.choice(image_list, number_of_samples, weights=sample_wts)  # picked images

                return picked_samples

            selected_samples = get_samples(strategy_params['image_list'], query,N)
            return selected_samples

        if name == 'uncertainty':

            # print("std dev", query.std(axis=1)[np.argsort(query.std(axis=1))])
            # return np.argsort(query.std(axis=1))[:N]

            tmp_query = np.array([query[i][0] for i in range(len(query))])
            difference_array = np.absolute(tmp_query - 0.5)
            # print("index", difference_array.argsort()[:N])
            # uncertainty = [query[i][0] for i in index]
            return difference_array.argsort()[:N]

            # uncertainty = []
            # for i in range(len(query)):
            #     if query[i][0] >= 0.1 and query[i][0] <= 0.9:
            #         uncertainty.append(i)
            # uncertainty = np.array(uncertainty)
            # if len(uncertainty) > N:
            #     return uncertainty[:N]
            # else:
            #     return uncertainty
        elif name == 'uncertainty_balanced':

            tmp_query = np.array([query[i][0] for i in range(len(query))])
            difference_array = (tmp_query - 0.5)
            sorted_diff = difference_array.argsort()

            # finding index of element closest to zero
            idx_0 = np.absolute(difference_array).argsort()[0]
            # idx_0 = sorted_diff[0]
            # for idx in sorted_diff:
            #     if abs(difference_array[idx]) < abs(difference_array[idx_0]):
            #         idx_0 = idx

            # finding where idx is in sorted_diff
            idx_sorted_0 = list(sorted_diff).index(idx_0)

            print("tmp_query- probs - sorted", sorted(tmp_query))
            print("diff arr ", difference_array)
            print("diff arr sorted ", np.sort(difference_array))
            print("middle value idx", idx_sorted_0)
            print("middle value ", difference_array[idx_sorted_0])

            N = (N - 1)
            # take n/2 from either side of index_sorted_0, less ele
            print("splitting idices", (idx_sorted_0 - (N // 2) - 1), idx_sorted_0)
            print("splitting idices", idx_sorted_0 + 1, (idx_sorted_0 + (N // 2)))
            tmp1 = sorted_diff[(idx_sorted_0 - (N // 2)):idx_sorted_0]
            tmp2 = sorted_diff[idx_sorted_0 + 1: (idx_sorted_0 + (N // 2)) + 1]
            print("tmp1", tmp1)
            print("tmp2", tmp2)
            print(f"uncertainity balanced: {len(tmp1)} + {len(tmp2)} = {len(tmp1) + len(tmp2)}")

            # returning the corresponding indexes tmp1 + tmp2
            tmp3 = []
            for i in tmp1:
                tmp3.append(i)
            for i in tmp2:
                tmp3.append(i)
            tmp3.append((idx_0))
            return tmp3

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

    def get_images_to_label_offline(self, model, sampling_strat, sample_size, strategy_params, device,path):
        # Load stuff
        model.eval()
        if device == "cuda":
            model.cuda()



        # dataset = self.embeddings
        # image_paths = self.images_path
        #
        # # Forward Pass
        # with torch.no_grad():
        #     bs = 128
        #     if len(dataset) < bs:
        #         bs = 1
        #     loader = DataLoader(dataset, batch_size=bs, shuffle=False)
        #     model_predictions = []
        #     for batch in tqdm(loader):
        #         if device == "cuda":
        #             x = torch.FloatTensor(batch).to(device)  # torch.cuda.FloatTensor(batch)
        #         else:
        #             x = torch.FloatTensor(batch)
        #         x = x.view(x.size(0), -1)
        #         predictions = model.linear_model(x)
        #         model_predictions.extend(predictions.cpu().detach().numpy())

        unlabled_probablites = []
        # model.eval()
        # model.cuda()
        image_size = 256

        def to_tensor(pil):
            return torch.tensor(np.array(pil)).permute(2, 0, 1).float()

        t = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.Lambda(to_tensor)
        ])
        dataset = ImageFolder(path, transform=t)
        img_paths = [i[0] for i in dataset.imgs]
        with torch.no_grad():
            bs = 128
            if len(dataset) < bs:
                bs = 1
            loader = DataLoader(dataset, batch_size=bs, shuffle=False)
            for batch in tqdm(loader):
                x = batch[0].cuda()
                # predictions = model(x)
                feats = model.encoder(x)[-1]
                feats = feats.view(feats.size(0), -1)
                predictions = model.linear_model(feats)
                unlabled_probablites.extend(predictions.detach().cpu().numpy())




        # Strategy
        #model_predictions = np.array(model_predictions)
        unlabled_probablites = np.array(unlabled_probablites)
        # print("m_predic",model_predictions)
        #subset = self.strategy(sampling_strat, model_predictions, sample_size, strategy_params)
        strategy_params['image_list']= img_paths
        subset = self.strategy(sampling_strat, unlabled_probablites, sample_size, strategy_params)
        # print("subset",subset)

        # Stuff to return
        #strategy_embeddings = np.array([i for i in dataset])[subset]
        strategy_embeddings = []
        #strategy_images = np.array([i for i in image_paths])[subset]

        if(sampling_strat=="gaussian"):
            strategy_images = subset
        else:
            strategy_images = np.array([i for i in img_paths])[subset]
        # print("model pred", model_predictions)
        # print("strategy_images", strategy_images)

        return strategy_embeddings, strategy_images

    def get_probablities(self, DATA_PATH, model, prob, image_size, paths=False):

        # model = CLASSIFIER.CLASSIFIER.load_from_checkpoint(MODEL_PATH)
        # pass as data_path
        # evaluation data_p / pos => all positive images => how many imgs model thinks is pos with accuracy of 0.8 => pos_class_confidence
        # evaluation data_n /neg  =>  all neg images => same => neg_class_confidence_0.8
        # pos,neg => 0.8 0.5  = prob

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
        img_paths = [i[0] for i in dataset.imgs]
        with torch.no_grad():
            bs = 128
            if len(dataset) < bs:
                bs = 1
            loader = DataLoader(dataset, batch_size=bs, shuffle=False)
            for batch in tqdm(loader):
                x = batch[0].cuda()
                # predictions = model(x)
                feats = model.encoder(x)[-1]
                feats = feats.view(feats.size(0), -1)
                predictions = model.linear_model(feats)
                unlabled_probablites.extend(predictions.detach().cpu().numpy())
        if paths:
            return img_paths, unlabled_probablites
        return unlabled_probablites

        # positive_predictions = np.array([unlabled_probablites[i][1] for i in range(len(unlabled_probablites))])
        # # count = 0
        # # for i in positive_predictions:
        # #     if i > prob:
        # #         count += 1 #TODO count , median
        # #median positive_predictions np.median(pos_predic)
        # return positive_predictions