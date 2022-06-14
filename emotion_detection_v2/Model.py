from transformers import AutoTokenizer,  AutoModelForSequenceClassification, AutoConfig
from torch.utils.data import DataLoader, Dataset
from typing import Optional, Dict, List, Tuple
import pytorch_lightning as pl
from . import Data




class EmotionPrediction(pl.LightningModule):
    def __init__(self, params, logger):
        super().__init__()
        self.args = params
        self.logging = logger
        if self.args.from_pretrained is not None or args.resume_ckpt is not None: ## TODO check if this is true with resume_ckpt
            self._set_config()
            self._load_pretrained()

        self.train_dataloader_object = self.val_dataloader_object = self.test_dataloader_object = None
        self.current_checkpoint =0
        self.best_checkpoint = None
        self.best_metric = 10000 if self.args.early_stopping_metric == 'vloss' else 0 ## keep track of best dev value of whatever metric is used in early stopping callback
        self.num_not_improved = 0
        self.save_hyperparameters()

    def _load_pretrained(self):
        self.sentence_classifier_model =  AutoModelForSequenceClassification.from_pretrained(self.args.from_pretrained, config=self.config, cache_dir=self.args.cache_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.tokenizer, use_fast=True, cache_dir=self.args.cache_dir)

    def _set_config(self):
        self.config = AutoConfig.from_pretrained(self.args.from_pretrained)
        self.config.attention_dropout = self.args.attention_dropout
        self.config.dropout = self.args.dropout
        self.config.activation_dropout = self.args.activation_dropout
        if self.config.use_cache and self.args.grad_ckpt:
            self.config.use_cache = False

    def get_tokenizer(self,):
        return self.tokenizer

    def set_datasets(self,
                     train_set: Data.EmotionDataset,
                     dev_set:  Data.EmotionDataset,
                     test_set:  Optional[Data.EmotionDataset]=None ) :
        self.train_set = train_set
        self.dev_set = dev_set
        if test_set is not None:
            self.test_set = test_set

    def get_attention_mask(input_ids, pad_token_id):
        attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=input_ids.device)
        attention_mask[input_ids == pad_token_id] = 0
        return attention_mask

    def _get_dataloader(self, current_dataloader, split_name, is_train):
        if current_dataloader is not None:
            return current_dataloader

        if split_name == "train":
            dataset = self.train_set
        elif split_name == "dev":
            dataset = self.dev_set
        elif split_name == "test":
            dataset = self.test_set
        else:
            self.logging.log(f"Invalid split name: {split_name}")

        #print(self.trainer._accelerator_connector.strategy)
        sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=is_train)

        return DataLoader(dataset, batch_size=self.args.batch_size, shuffle=(sampler is None),
                          num_workers=self.args.num_workers, sampler=sampler,
                          collate_fn=EmotionDataset.collate_fn)

    def train_dataloader(self):
        self.train_dataloader_object = self._get_dataloader(self.train_dataloader_object, 'train', is_train=True)
        return self.train_dataloader_object

    def val_dataloader(self):
        self.val_dataloader_object = self._get_dataloader(self.val_dataloader_object, 'val', is_train=False)
        return self.val_dataloader_object

    def test_dataloader(self):
        self.test_dataloader_object = self._get_dataloader(self.test_dataloader_object, 'test', is_train=False)
        return self.test_dataloader_object


    def forward(self, input_ids, labels):
        attention_mask = self.get_attention_mask(input_ids, self.tokenizer.pad_token_id)
        print("in ids ", input_ids)
        print("attention mask ", attention_mask)
        print("labels ", labels)
        exit(0)

    ### TODO

