import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer

class POGERDataset(Dataset):
    def __init__(self, path, pretrain_model, max_len, k):
        self.max_len = max_len
        self.k = k
        self.tokenizer = RobertaTokenizer.from_pretrained(pretrain_model)

        with open(path, 'r') as f:
            lines = f.readlines()
        self.data = [json.loads(line) for line in lines]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        inputs = self.tokenizer(item['text'], max_length=self.max_len, padding='max_length', truncation=True)

        est_prob = torch.tensor(item['est_prob_list'])  # (n_feat, 10)
        est_prob = est_prob.t()  # (10, n_feat)
        est_prob = torch.cat([est_prob, torch.zeros(self.k - est_prob.shape[0], est_prob.shape[1])], dim=0)

        target_roberta_idx = torch.tensor(item['target_roberta_idx'])  # (10, )
        target_roberta_idx = torch.cat([target_roberta_idx, torch.zeros(self.k - target_roberta_idx.shape[0])], dim=0).long()

        return {
            'input_ids': torch.tensor(inputs['input_ids']),
            'attention_mask': torch.tensor(inputs['attention_mask']),
            'est_prob': est_prob,
            'target_roberta_idx': target_roberta_idx,
            'label': torch.tensor(item['label_int']),
        }

class POGERMixDataset(Dataset):
    def __init__(self, path, pretrain_model, max_len, k):
        self.max_len = max_len
        self.k = k
        self.tokenizer = RobertaTokenizer.from_pretrained(pretrain_model)

        with open(path, 'r') as f:
            lines = f.readlines()
        self.data = [json.loads(line) for line in lines]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        inputs = self.tokenizer(item['text'], max_length=self.max_len, padding='max_length', truncation=True)

        mix_prob = torch.tensor(item['mix_prob_list'])  # (n_feat, seq_len)
        mix_prob = mix_prob.t()  # (seq_len, n_feat)
        # fill mix_prob to (max_len, n_feat) and get bool mask
        mix_prob = mix_prob[:self.max_len]
        mix_prob_mask = torch.cat([torch.zeros(mix_prob.shape[0]), torch.ones(self.max_len - mix_prob.shape[0])], dim=0).bool()
        mix_prob = torch.cat([mix_prob, torch.zeros(self.max_len - mix_prob.shape[0], mix_prob.shape[1])], dim=0)

        target_roberta_idx = torch.tensor(item['target_roberta_idx'])  # (10, )
        target_roberta_idx = torch.cat([target_roberta_idx, torch.zeros(10 - target_roberta_idx.shape[0])], dim=0).long()

        target_prob_idx = torch.tensor(item['target_prob_idx'])  # (10, )
        target_prob_idx = torch.cat([target_prob_idx, torch.zeros(self.k - target_prob_idx.shape[0])], dim=0).long()

        return {
            'input_ids': torch.tensor(inputs['input_ids']),
            'attention_mask': torch.tensor(inputs['attention_mask']),
            'mix_prob': mix_prob,
            'mix_prob_mask': mix_prob_mask,
            'target_roberta_idx': target_roberta_idx,
            'target_prob_idx': target_prob_idx,
            'label': torch.tensor(item['label_int']),
        }

def get_dataloader(model_name, data_dir, pretrain_model, batch_size, max_len, k):
    if model_name == 'poger' or model_name == 'poger_wo_context':
        train_dataset = POGERDataset(os.path.join(data_dir, 'train_poger_feature.jsonl'), pretrain_model, max_len, k)
        test_dataset = POGERDataset(os.path.join(data_dir, 'test_poger_feature.jsonl'), pretrain_model, max_len, k)
    elif model_name == 'poger_mix' or model_name == 'poger_mix_wo_context':
        train_dataset = POGERMixDataset(os.path.join(data_dir, 'train_poger_mix_feature.jsonl'), pretrain_model, max_len, k)
        test_dataset = POGERMixDataset(os.path.join(data_dir, 'test_poger_mix_feature.jsonl'), pretrain_model, max_len, k)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_dataloader, test_dataloader
