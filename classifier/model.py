import os
from transformers import AutoModel
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from sklearn.metrics import classification_report

from classifier.bert import BERT_Arch



class Model:
    def __init__(self, model_version, hidden_size, device, lr=1e-5, model = None):
        self.model_version = model_version
        self.hidden_size = hidden_size
        self.device = device
        self.lr = lr
        self.loss_fn = nn.CrossEntropyLoss()
        self.model_path = './var/model'
        self.model_file = 'weights.pt'

        if model:
            hidden_size = model.config.hidden_size
            self.model = BERT_Arch(model, hidden_size)
        else:

            self.model = AutoModel.from_pretrained(model_version)

            for param in self.model.parameters():
                param.requires_grad = False

            self.model = BERT_Arch(self.model, hidden_size)

        self.model.to(device)

        self.optimizer = AdamW(self.model.parameters(), lr=self.lr)

    def train(self, train_dataloader, eval_dataloader, epochs):

        best_valid_loss = float('inf')

        train_losses = []
        valid_losses = []

        for epoch in tqdm(range(epochs)):

            print('\n Epoch {:} / {:}'.format(epoch + 1, epochs))

            train_loss, _ = self.model.train_step(train_dataloader, self.device, self.loss_fn, self.optimizer)
            valid_loss, _ = self.model.evaluate(eval_dataloader, self.device, self.loss_fn)

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(self.model.state_dict(), os.path.join(self.model_path, self.model_file))

            train_losses.append(train_loss)
            valid_losses.append(valid_loss)

            print(f'\nTraining Loss: {train_loss:.3f}')
            print(f'Validation Loss: {valid_loss:.3f}')

    def predict(self, test_seq, test_mask):
        with torch.no_grad():
            return np.argmax(
                self.model(test_seq.to(self.device), test_mask.to(self.device)).detach().cpu().numpy(),
                axis=1
            )

    def prepare_classification_report(self, test_y, preds):
        return classification_report(test_y, preds)
