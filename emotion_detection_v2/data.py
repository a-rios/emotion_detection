from transformers import AutoTokenizer
from torch.utils.data import DataLoader, Dataset
from typing import Optional, Dict, List, Tuple
import pandas as pd
import torch
import math
import numpy as np


class EmotionDataset(Dataset):
    def __init__(self,
                 in_file: str,
                 tokenizer: AutoTokenizer,
                 file_format: str,
                 label_name: str,
                 utterance_name: str,
                 split_name: str,
                 prediction_only: Optional[bool]=False,
                 emotions: Optional[dict]=None,
                 max_len: Optional[int]=None):
        self.tokenizer = tokenizer
        self.file_format = file_format
        self.utterance_name = utterance_name
        self.label_name = label_name
        self.split_name =split_name # train, dev, test

        if self.file_format == "json":
            self.df = pd.read_json(in_file)
        elif self.file_format == "csv":
            self.df = pd.read_csv(in_file)

        # only get emotions from training set
        self.emotions = emotions if emotions is not None else {e:i for  i,e in  enumerate(self.df[self.label_name].unique()) }
        self.labels = [l for l in self.df[label_name]]

        # only get max len from training set
        if max_len is None:
            self.sources = [self.tokenize_input(utterance) for utterance in self.df[utterance_name]] # no truncation
            self.max_len =  self._calculate_max_length(self.sources)
        else:
            self.max_len = max_len
            self.sources = [self.tokenize_input(utterance, self.max_len) for utterance in self.df[utterance_name]]

    def __len__(self):
        return len(self.sources)


    def get_max_len(self,):
        return self.max_len

    def get_emotions(self,):
        return self.emotions

    def tokenize_input(self,
                       utterance: str,
                       max_len: Optional[int]=None):
        if max_len is None:
            return self.tokenizer.encode(utterance, truncation=False, add_special_tokens=True)
        else:
            return self.tokenizer.encode(utterance, truncation=True, add_special_tokens=True, max_length=max_len)

    def _calculate_max_length(self,
                       utterances: List[List[int]]):
        return max([len(l) for l in utterances])


    def __getitem__(self, idx):
        input_ids = torch.tensor(self.sources[idx])
        label_id = self.emotions[self.labels[idx]]
        labels = torch.zeros(len(self.emotions),)
        labels[label_id] =1
        return input_ids, labels

    @staticmethod
    def collate_fn(batch, pad_token_id):
        input_ids, labels = list(zip(*batch))
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id)
        labels = torch.stack(labels)
        return input_ids, labels

    def calc_loss_weights(self,
                          rate: float,
                          bww: bool = False):
        label_count = {}
        for l in self.emotions.keys():
            label_count[l] =self.df[self.label_name].value_counts()[l]

        min_l = float(min([ label_count[w] for w in self.emotions]))
        if bww:
            weight = [math.pow(min_l / label_count[k], rate) if k in self.emotions else 0 for k,v in label_count.items()]
        else:
            weight = [math.pow(1, rate) if k in self.emotions else 0 for k,v in label_count.items()]
        weight = np.array(weight)
        weight /= np.sum(weight)
        weight = torch.from_numpy(weight).float()
        return weight

    def calc_class_weights(self,
                           rate: float):
        label_count = {}
        for l in self.emotions.keys():
            label_count[l] =self.df[self.label_name].value_counts()[l]


        weight = [(v / sum(label_count.values())) for k,v in label_count.items()]
        weight = np.array(weight)
        weight = torch.from_numpy(weight).float()
        return weight



