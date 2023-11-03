import numpy as np
import torch
import torch.nn as nn


class BERT_Arch(nn.Module):
    def __init__(self, bert, hidden_size):
        super(BERT_Arch, self).__init__()
        self.bert = bert
        self.dropout = nn.Dropout(0.1)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(hidden_size, 512)
        self.fc2 = nn.Linear(512, 2)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, sent_id, mask):
        _, cls_hs = self.bert(sent_id, attention_mask=mask, return_dict=False)
        x = self.fc1(cls_hs)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x

    def train_step(self, dataloader, device, loss_fn, optimizer):
        self.train()
        total_loss, total_accuracy = 0, 0
        total_preds = []
        for step, batch in enumerate(dataloader):
            batch = [r.to(device) for r in batch]
            sent_id, mask, labels = batch
            self.zero_grad()
            preds = self(sent_id, mask)
            loss = loss_fn(preds, labels)
            total_loss = total_loss + loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
            optimizer.step()
            preds = preds.detach().cpu().numpy()
            total_preds.append(preds)

        avg_loss = total_loss / len(dataloader)
        total_preds = np.concatenate(total_preds, axis=0)

        return avg_loss, total_preds

    def evaluate(self, dataloader, device, loss_fn):
        self.eval()
        total_loss, total_accuracy = 0, 0
        total_preds = []

        for step, batch in enumerate(dataloader):
            batch = [t.to(device) for t in batch]
            sent_id, mask, labels = batch

            with torch.no_grad():
                preds = self(sent_id, mask)
                loss = loss_fn(preds, labels)
                total_loss = total_loss + loss.item()
                preds = preds.detach().cpu().numpy()
                total_preds.append(preds)

        avg_loss = total_loss / len(dataloader)
        total_preds = np.concatenate(total_preds, axis=0)

        return avg_loss, total_preds
