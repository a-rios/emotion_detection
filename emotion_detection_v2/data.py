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
                 remove_unaligned: bool = True,
                 no_labels: Optional[bool]=False,
                 emotions: Optional[dict]=None,
                 max_len: Optional[int]=None,
                 save_texts: Optional[bool]=False,
                 csv_delimiter: Optional[str]=","):
        self.tokenizer = tokenizer
        self.file_format = file_format
        self.utterance_name = utterance_name
        self.label_name = label_name
        self.split_name = split_name # train, dev, test
        self.no_labels = no_labels

        if self.file_format == "json":
            self.df = pd.read_json(in_file)
        elif self.file_format == "csv":
            self.df = pd.read_csv(in_file, sep=csv_delimiter)

        # only get emotions from training set
        self.emotions = emotions if emotions is not None else {e:i for  i,e in  enumerate(self.df[self.label_name].unique()) }

        if self.no_labels: # only for testing
            self.sources = [self.tokenize_input(utterance) for utterance in self.df[utterance_name] ] # no truncation
            self.max_len =  self._calculate_max_length(self.sources)
            if save_texts:
                self.texts = [utterance for utterance in self.df[utterance_name]]
        else:
            # only get max len from training set
            if remove_unaligned:
                if max_len is None:
                    self.sources, self.labels = zip(*[(self.tokenize_input(utterance), label) for utterance, label in zip(self.df[utterance_name], self.df[label_name])  if utterance != "NOT FOUND"]) # no truncation
                    self.max_len =  self._calculate_max_length(self.sources)
                    if save_texts:
                        self.texts = [utterance for utterance in self.df[utterance_name] if utterance != "NOT FOUND"]
                else:
                    self.max_len = max_len
                    self.sources, self.labels = zip(*[(self.tokenize_input(utterance, self.max_len), label) for utterance, label in zip (self.df[utterance_name],self.df[label_name]) if utterance != "NOT FOUND"  ])
                    if save_texts:
                        self.texts = [utterance for utterance in self.df[utterance_name] if utterance != "NOT FOUND"]
            else:
                if max_len is None:
                    self.sources, self.labels = zip(*[(self.tokenize_input(utterance), label) for utterance, label in zip(self.df[utterance_name], self.df[label_name]) ]) # no truncation
                    self.max_len =  self._calculate_max_length(self.sources)
                    if save_texts:
                        self.texts = [utterance for utterance in self.df[utterance_name]]
                else:
                    self.max_len = max_len
                    self.sources, self.labels = zip(*[(self.tokenize_input(utterance, self.max_len), label) for utterance, label in zip (self.df[utterance_name],self.df[label_name]) ])
                    if save_texts:
                        self.texts = [utterance for utterance in self.df[utterance_name]]

            print(f"Frequency of labels in {split_name}: ")
            for e in self.emotions.keys():
                print(f"({self.emotions[e]}:{e}, {self.labels.count(e)})", end=" ")
            print()

    def __len__(self):
        return len(self.sources)


    def get_max_len(self,):
        return self.max_len

    def get_emotions(self,):
        return self.emotions

    def get_labels(self,):
        return self.labels

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
        labels = None
        if not self.no_labels:
            label_id = self.emotions[self.labels[idx]]
            labels = torch.tensor([label_id])
        text = None
        if hasattr(self, "texts"):
            text = str(idx) +":" + self.texts[idx] if self.texts is not None else None # id: text
        return input_ids, labels, text

    @staticmethod
    def collate_fn(batch, pad_token_id):
        input_ids, labels, text = list(zip(*batch))
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id)
        if not None in labels:
            labels = torch.stack(labels)
            labels = labels.squeeze(1)
        return input_ids, labels, text

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



