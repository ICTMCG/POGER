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

class Attention(torch.nn.Module):
    """
    Compute 'Scaled Dot Product Attention
    """

    def forward(self, query, key, value, mask=None, dropout=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(query.size(-1))

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        p_attn = F.softmax(scores, dim=-1)

        if dropout is not None:
            p_attn = dropout(p_attn)

        return torch.matmul(p_attn, value), p_attn

class MultiHeadedAttention(torch.nn.Module):
    """
    Take in model size and number of heads.
    """

    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0

        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h

        self.linear_layers = torch.nn.ModuleList([torch.nn.Linear(d_model, d_model) for _ in range(3)])
        self.output_linear = torch.nn.Linear(d_model, d_model)
        self.attention = Attention()

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        if mask is not None:
            mask = mask.repeat(1, self.h, 1, 1)
        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linear_layers, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)
        # print('x shape after self attention: {}'.format(x.shape))

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)

        return self.output_linear(x), attn

class SelfAttentionFeatureExtract(torch.nn.Module):
    def __init__(self, multi_head_num, input_size, output_size=None):
        super(SelfAttentionFeatureExtract, self).__init__()
        self.attention = MultiHeadedAttention(multi_head_num, input_size)
        # self.out_layer = torch.nn.Linear(input_size, output_size)
    def forward(self, inputs, query, mask=None):
        if mask is not None:
            mask = mask.view(mask.size(0), 1, 1, mask.size(-1))

        feature, attn = self.attention(query=query,
                                 value=inputs,
                                 key=inputs,
                                 mask=mask
                                 )
        return feature, attn

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

        self.reducer = MLP(768, [384], embedding_size, dropout)
        self.cross_attention_context = SelfAttentionFeatureExtract(1, embedding_size)
        self.cross_attention_prob = SelfAttentionFeatureExtract(1, embedding_size)
        self.classifier = MLP(embedding_size * 2, [128, 32], nclasses, dropout)

    def conv_feat_extract(self, x):
        out = self.conv(x)
        out = out.transpose(1, 2)
        return out

    def forward(self, prob_feature, sem_feature, target_roberta_idx):
        # extract sem_feature of target_roberta_idx
        context_feature = sem_feature.gather(1, target_roberta_idx.unsqueeze(-1).expand(-1, -1, sem_feature.shape[-1]))  # (batch_size, 10, 768)
        # reduce sem_feature
        context_feature = self.reducer(context_feature)  # (batch_size, 10, embedding_size)
        # cnn
        prob_feature = prob_feature.transpose(1, 2)
        prob_feature = torch.cat([self.conv_feat_extract(prob_feature[:, i:i+1, :]) for i in range(self.nfeat)], dim=2)  # (batch_size, 10, embedding_size)

        prob_feature = prob_feature + self.position_encoding.cuda()
        prob_feature = self.norm(prob_feature)
        prob_feature = self.encoder(prob_feature)
        prob_feature = self.dropout(prob_feature)

        # reweight prob_feature
        prob_feature, _ = self.cross_attention_prob(prob_feature, context_feature)  # (batch_size, 10, embedding_size)
        # reweight context_feature
        context_feature, _ = self.cross_attention_context(context_feature, prob_feature)  # (batch_size, 10, embedding_size)
        # concat
        merged = torch.cat([prob_feature, context_feature], dim=-1)  # (batch_size, 10, embedding_size * 2)
        # classify
        merged = self.classifier(merged)  # (batch_size, 10, nclasses)
        # mean
        output = merged.mean(dim=1)  # (batch_size, nclasses)
        return output

class Trainer(TrainerBase):
    def __init__(self, device, pretrain_model, train_dataloader, test_dataloader, epoch, lr, model_save_path, n_classes, n_feat, k):
        super(Trainer, self).__init__(device, pretrain_model, train_dataloader, test_dataloader, epoch, lr, model_save_path, n_classes)
        self.pretrain = RobertaModel.from_pretrained(pretrain_model).to(device)
        self.model_save_path = model_save_path
        self.model = Model(nfeat=n_feat, nclasses=n_classes, k=k).to(device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def get_loss(self, batch):
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        sem_feature = self.pretrain(input_ids, attention_mask).last_hidden_state.detach()
        prob_feature = batch['est_prob'].to(self.device)
        target_roberta_idx = batch['target_roberta_idx'].to(self.device)
        label = batch['label'].to(self.device)

        output = self.model(prob_feature, sem_feature, target_roberta_idx)
        loss = self.criterion(output, label)
        return loss

    def get_output(self, batch):
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        sem_feature = self.pretrain(input_ids, attention_mask).last_hidden_state.detach()
        prob_feature = batch['est_prob'].to(self.device)
        target_roberta_idx = batch['target_roberta_idx'].to(self.device)
        with torch.no_grad():
            output = self.model(prob_feature, sem_feature, target_roberta_idx)
        return output
