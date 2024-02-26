import os
import math
import torch
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch import nn, optim
from transformers import RobertaModel
from typing import List, Tuple

from utils.functions import MLP
from utils.trainer import Trainer as TrainerBase

class ConvFeatureExtractionModel(nn.Module):

    def __init__(
        self,
        conv_layers: List[Tuple[int, int, int]],
        conv_dropout: float = 0.0,
        conv_bias: bool = False,
    ):
        super().__init__()

        def block(n_in, n_out, k, stride=1, conv_bias=False):
            padding = k // 2
            return nn.Sequential(
                nn.Conv1d(in_channels=n_in, out_channels=n_out, kernel_size=k, stride=stride, padding=padding, bias=conv_bias),
                nn.Dropout(conv_dropout),
                nn.ReLU()
            )

        in_d = 1
        self.conv_layers = nn.ModuleList()
        for _, cl in enumerate(conv_layers):
            assert len(cl) == 3, "invalid conv definition: " + str(cl)
            (dim, k, stride) = cl

            self.conv_layers.append(
                block(in_d, dim, k, stride=stride, conv_bias=conv_bias))
            in_d = dim

    def forward(self, x):
        # x = x.unsqueeze(1)
        for conv in self.conv_layers:
            x = conv(x)
        return x

class Model(nn.Module):

    def __init__(self, nfeat, nclasses, dropout=0.2, k=10):
        super(Model, self).__init__()
        self.nfeat = nfeat

        feature_enc_layers = [(64, 5, 1)] + [(128, 3, 1)] * 3 + [(64, 3, 1)]
        self.conv = ConvFeatureExtractionModel(
            conv_layers=feature_enc_layers,
            conv_dropout=0.0,
            conv_bias=False,
        )
        embedding_size = nfeat *64

        self.encoder_layer = TransformerEncoderLayer(
            d_model=embedding_size,
            nhead=4,
            dim_feedforward=256,
            dropout=0.1,
            batch_first=True)
        self.encoder = TransformerEncoder(encoder_layer=self.encoder_layer,
                                            num_layers=2)
        seq_len = k
        self.position_encoding = torch.zeros((seq_len, embedding_size))
        for pos in range(seq_len):
            for i in range(0, embedding_size, 2):
                self.position_encoding[pos, i] = torch.sin(
                    torch.tensor(pos / (10000**((2 * i) / embedding_size))))
                self.position_encoding[pos, i + 1] = torch.cos(
                    torch.tensor(pos / (10000**((2 *
                                                 (i + 1)) / embedding_size))))
        
        self.norm = nn.LayerNorm(embedding_size)
        self.dropout = nn.Dropout(0.1)

        self.classifier = MLP(embedding_size, [128, 32], nclasses, dropout)

    def conv_feat_extract(self, x):
        out = self.conv(x)
        out = out.transpose(1, 2)
        return out

    def forward(self, prob_feature):
        prob_feature = prob_feature.transpose(1, 2)
        prob_feature = torch.cat([self.conv_feat_extract(prob_feature[:, i:i+1, :]) for i in range(self.nfeat)], dim=2)  # (batch_size, 10, embedding_size)

        prob_feature = prob_feature + self.position_encoding.cuda()
        prob_feature = self.norm(prob_feature)
        prob_feature = self.encoder(prob_feature)
        prob_feature = self.dropout(prob_feature)

        # classify
        output = self.classifier(prob_feature)  # (batch_size, 10, nclasses)
        # mean
        output = output.mean(dim=1)  # (batch_size, nclasses)
        return output

class Trainer(TrainerBase):
    def __init__(self, device, pretrain_model, train_dataloader, test_dataloader, epoch, lr, model_save_path, n_classes, n_feat, k):
        super(Trainer, self).__init__(device, pretrain_model, train_dataloader, test_dataloader, epoch, lr, model_save_path, n_classes)
        self.model_save_path = model_save_path
        self.model = Model(nfeat=n_feat, nclasses=n_classes, k=k).to(device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def get_loss(self, batch):
        prob_feature = batch['est_prob'].to(self.device)
        label = batch['label'].to(self.device)

        output = self.model(prob_feature)
        loss = self.criterion(output, label)
        return loss

    def get_output(self, batch):
        prob_feature = batch['est_prob'].to(self.device)
        with torch.no_grad():
            output = self.model(prob_feature)
        return output
