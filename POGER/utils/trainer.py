import torch
import numpy as np
from tqdm import tqdm

from utils.utils import Averager, metrics

class Trainer:
    def __init__(self, device, pretrain_model, train_dataloader, test_dataloader, epoch, lr, model_save_path, n_classes):
        self.device = device
        self.epoch = epoch
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.n_classes = n_classes
        # Need to define self.model_save_path, self.model, self.criterion and self.optimizer in derived class

    def get_loss(self, batch):
        # Need to be implemented in derived class
        return None

    def get_output(self, batch):
        # Need to be implemented in derived class
        return None

    def train(self):
        for epoch in range(self.epoch):
            print('----epoch %d----' % (epoch+1))
            self.model.train()
            avg_loss = Averager()
            for i, batch in enumerate(tqdm(self.train_dataloader)):
                self.optimizer.zero_grad()
                loss = self.get_loss(batch)
                loss.backward()
                self.optimizer.step()
                avg_loss.add(loss.item())
            results = self.test(self.test_dataloader)
            print('epoch %d: loss = %.4f, acc = %.4f, f1 = %.4f, auc_ovo = %.4f' % (epoch+1, avg_loss.get(), results['accuracy'], results['f1'], results['auc_ovo']))
            print('P/R per class: ', end='')
            for i in range(self.n_classes):
                print('%.2f/%.2f ' % (results['precision'][i] * 100, results['recall'][i] * 100), end='')
            print()
            print('F1 per class: ', end='')
            for i in range(self.n_classes):
                print('%.2f ' % (results['detail_f1'][i] * 100), end='')
            print()

            torch.save(self.model.state_dict(), self.model_save_path)

    def test(self, dataloader):
        self.model.eval()
        y_true = torch.empty(0)
        y_score = torch.empty((0, self.n_classes))
        for i, batch in enumerate(tqdm(dataloader)):
            output = self.get_output(batch).cpu()
            output = torch.softmax(output, dim=1)
            y_score = torch.cat((y_score, output))
            y_true = torch.cat((y_true, batch['label']))

        results = metrics(y_true, y_score)
        return results
