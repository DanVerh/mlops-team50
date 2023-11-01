import requests
import tarfile
import os
from os import path
import pandas as pd
import re
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler



class Dataset:
    def __init__(self):
        self.data_path = './var/data'

    def download_commonsense_data(self, tar_file_url, tar_file):
        response = requests.get(tar_file_url, stream=True)
        tar_file_path = os.path.join(self.data_path, tar_file)
        if response.status_code == 200:
            with open(tar_file_path, 'wb') as f:
                f.write(response.raw.read())

        files_to_extract = ['ethics/commonsense/cm_train.csv',
                            'ethics/commonsense/cm_test.csv',
                            'ethics/commonsense/cm_test_hard.csv']
        with tarfile.open(tar_file_path, 'r') as tar:
            for file_info in tar.getmembers():
                if file_info.name in files_to_extract:
                    tar.extract(file_info, self.data_path)

    def preprocess(self,
                   tar_file_url='https://people.eecs.berkeley.edu/~hendrycks/ethics.tar',
                   target_folder='commonsense'):

        tar_path = re.findall('/([^/]+)\.tar$', tar_file_url)[0]
        tar_file = f'{tar_path}.tar'
        dataset_path = os.path.join(self.data_path, tar_path, target_folder)
        if not path.exists(dataset_path):
            self.download_commonsense_data(tar_file_url, tar_file)

        data = (pd.concat([pd.read_csv(os.path.join(dataset_path, file)) for file in os.listdir(dataset_path)])
                .sample(frac=1, random_state=42)
                .reset_index(drop=True)
                .dropna(subset='input', axis=0)
                .drop(['edited', 'is_short'], axis=1)
                .rename({'input': 'text'}, axis=1)
        )
        train, temp = train_test_split(data, test_size=0.2, random_state=42)
        val, test = train_test_split(temp, test_size=0.5, random_state=42)
        return {'train': train, 'val': val, 'test': test}

    def prepare_dataloaders(self, split, tokenizer, hidden_size, batch_size):
        data = self.preprocess()[split]

        dataset = TensorDataset(*self.prepare_tensors_from_df(data, tokenizer, hidden_size))
        if split == 'train':
            sampler = RandomSampler(dataset)
        else:
            sampler = SequentialSampler(dataset)
        return DataLoader(dataset=dataset, sampler=sampler, batch_size=batch_size)

    def prepare_tensors_from_df(self, data: pd.DataFrame, tokenizer, hidden_size):
        tokens = self.tokenize(data['text'].to_list(), tokenizer, hidden_size)
        seq = torch.tensor(tokens['input_ids'])
        mask = torch.tensor(tokens['attention_mask'])
        y = torch.tensor(data['label'].to_list())
        return seq, mask, y

    def tokenize(self, text, tokenizer, hidden_size):
        return tokenizer.batch_encode_plus(text,
                                           max_length=hidden_size,
                                           padding='max_length',
                                           # pad_to_max_length=True,
                                           truncation=True
        )

    def prepare_tensors(self, data: list[str], tokenizer, hidden_size):
        tokens = self.tokenize(data, tokenizer, hidden_size)
        seq = torch.tensor(tokens['input_ids'])
        mask = torch.tensor(tokens['attention_mask'])
        return seq, mask