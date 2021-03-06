import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import SGD
import pytorch_lightning as pl
from pytorch_lightning.metrics import Accuracy


class ClassifierModel(pl.LightningModule): #SSLFineTuner
    def __init__(self, parameters, encoder, linear_model):
        # Init
        super().__init__()
        self.model_params = parameters
        # Models
        self.encoder = encoder
        self.linear_model = linear_model

        #Encoder parameters
        self.embedding_size = self.model_params['encoder']['e_embedding_size']
        self.lr_encoder = float(self.model_params['encoder']['e_lr'])
        self.train_encoder = self.model_params['encoder']['train_encoder']

        #Parameters for Classifier model
        self.num_classes = self.model_params['classifier']['c_num_classes']
        self.hidden_dim = self.model_params['classifier']['c_hidden_dim']
        self.lr = float(self.model_params['classifier']['c_linear_lr'])
        self.dropout = self.model_params['classifier']['c_dropout']
        self.scheduler_type = self.model_params['classifier']['c_scheduler_type'] 
        self.gamma = self.model_params['classifier']['c_gamma']
        self.decay_epochs = self.model_params['classifier']['c_decay_epochs']
        self.weight_decay = float(self.model_params['classifier']['c_weight_decay'])
        self.final_lr = float(self.model_params['classifier']['c_final_lr'])
        self.momentum = self.model_params['classifier']['c_momentum']
        self.weights = self.model_params['classifier']['c_weights']
        if self.weights is not None:
            self.weights = torch.tensor([float(item) for item in self.weights.split(',')])
            self.weights = self.weights.cuda()
        else:
            self.weights = None

        #Training parameters
        self.epochs = self.model_params['training']['epochs']
        self.train_acc = Accuracy()
        self.val_acc = Accuracy(compute_on_step=False)

        #Other parameters
        self.seed = self.model_params['miscellaneous']['seed']
        self.cpus = self.model_params['miscellaneous']['cpus']


        self.save_hyperparameters()
    
    def configure_optimizers(self):
        if self.train_encoder:
            optimizer = SGD([
                    {'params': self.encoder.parameters(), 'lr': self.lr_encoder},
                    {'params': self.linear_model.parameters(), 'lr': self.lr}
                    ], momentum=self.momentum)
        else:
            optimizer = SGD([
                    {'params': self.encoder.parameters(), 'lr': 0},
                    {'params': self.linear_model.parameters(), 'lr': self.lr}
                    ], momentum=self.momentum)

        if self.scheduler_type == "step":
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, self.decay_epochs, gamma=self.gamma)
        elif self.scheduler_type == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.epochs, eta_min=self.final_lr)
        return [optimizer], [scheduler]

    def freeze_encoder(self):
        self.train_encoder = False
        for param in self.encoder.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self):
        self.train_encoder = True
        for param in self.encoder.parameters():
            param.requires_grad = True

    def forward(self, x):
        if self.train_encoder:
            feats = self.encoder(x)[-1]
            feats = feats.view(feats.size(0), -1)
            logits = self.linear_model(feats)
        else:
            logits = self.linear_model(x)
        return logits

    def shared_step(self, batch):
        x, y = batch
        logits = self.forward(x)
        loss = self.loss_fn(logits, y)
        return loss, logits, y

    def training_step(self, batch, batch_idx):
        loss, logits, y = self.shared_step(batch)
        acc = self.train_acc(logits, y)
        self.log('tloss', loss, prog_bar=True)
        self.log('tastep', acc, prog_bar=True)
        self.log('ta_epoch', self.train_acc)
        return loss

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            loss, logits, y = self.shared_step(batch)
            acc = self.val_acc(logits, y)
        acc = self.val_acc(logits, y)
        self.log('val_loss', loss, prog_bar=True, sync_dist=True)
        self.log('val_acc_epoch', self.val_acc, prog_bar=True)
        self.log('val_acc_epoch', self.val_acc, prog_bar=True)
        return loss

    def loss_fn(self, logits, labels):

        # return F.cross_entropy(logits, labels, weight=self.weights)

        labels = torch.unsqueeze(labels, 1)
        labels = labels.type(torch.cuda.FloatTensor)
        return F.binary_cross_entropy(logits, labels, weight = self.weights)
        #return F.binary_cross_entropy(logits, labels, pos_weight = self.weights)