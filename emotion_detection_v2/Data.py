from transformers import AutoTokenizer
from torch.utils.data import DataLoader, Dataset
from typing import Optional, Dict, List, Tuple
import pandas as pd



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
            df = pd.read_json(in_file)
        elif self.file_format == "csv":
            df = pd.read_csv(in_file)

        # only get emotions from training set
        self.emotions = emotions if emotions is not None else {e:i for  i,e in  enumerate(df[self.label_name].unique()) }
        self.labels = [l for l in df[label_name]]

        # only get max len from training set
        if max_len is None:
            self.sources = [self.tokenize_input(utterance) for utterance in df[utterance_name]] # no truncation
            self.max_len =  self._max_length(self.sources)
        else:
            self.max_len = max_len
            self.sources = [self.tokenize_input(utterance, self.max_len) for utterance in df[utterance_name]]



    def __len__(self):
        return len(self.sources)


    def _get_max_len(self,):
        return self.max_len

    def _get_emotions(self,):
        return self.emotions

    def tokenize_input(self,
                       utterance: str,
                       max_len: Optional[int]=None):
        if max_len is None:
            return self.tokenizer.encode(utterance, truncation=False, add_special_tokens=True)
        else:
            return self.tokenizer.encode(utterance, truncation=True, add_special_tokens=True, max_length=max_len)

    def _max_length(self,
                       utterances: List[List[int]]):
        return max([len(l) for l in utterances])


    def __getitem__(self, idx):
        input_ids = self.sources[idx]
        label_ids = self.labels[idx]
        return input_ids, label_ids

    @staticmethod
    def collate_fn(batch):
        pad_token_id = self.tokenizer.pad_token_id
        input_ids, output_ids = list(zip(*batch))
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id)
        return input_ids, output_ids



