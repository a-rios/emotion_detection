from transformers import AutoTokenizer,  AutoModelForSequenceClassification, AutoConfig
from torch.utils.data import DataLoader, Dataset
from typing import Optional, Dict, List, Tuple
import pytorch_lightning as pl
from .data import EmotionDataset
import torch
import numpy as np
from functools import partial
import torch.nn.functional as F
from . import metrics

class EmotionPrediction(pl.LightningModule):
    def __init__(self, params):
        super().__init__()
        self.args = params
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
        self.config = AutoConfig.from_pretrained(self.args.from_pretrained, problem_type="single_label_classification", num_labels=self.args.num_classes)
        self.config.classifier_dropout=0.1

       #self.config.attention_dropout = self.args.attention_dropout
        #self.config.dropout = self.args.dropout
        #self.config.activation_dropout = self.args.activation_dropout
        if self.config.use_cache and self.args.grad_ckpt:
            self.config.use_cache = False

    def get_tokenizer(self,):
        return self.tokenizer

    def set_datasets(self,
                     train_set: EmotionDataset,
                     dev_set:  EmotionDataset,
                     test_set:  Optional[EmotionDataset]=None ) :
        self.train_set = train_set
        self.dev_set = dev_set
        if test_set is not None:
            self.test_set = test_set

        # set class/loss weights according to frequencies in train set
        self.loss_weights  = self.train_set.calc_loss_weights(rate=self.args.weight_rate, bww=self.args.balanced_weight_warming)
        self.class_weights = self.train_set.calc_class_weights(rate=self.args.weight_rate) # TODO check weights, seem off

    def get_attention_mask(self, input_ids):
        attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=input_ids.device)
        attention_mask[input_ids == self.tokenizer.pad_token_id] = 0
        return attention_mask

    def _get_dataloader(self, current_dataloader, split_name, is_train):
        if current_dataloader is not None:
            return current_dataloader

        if split_name == "train":
            dataset = self.train_set
        elif split_name == "val":
            dataset = self.dev_set
        elif split_name == "test":
            dataset = self.test_set
        else:
            self.log(f"Invalid split name: {split_name}")

        sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=is_train)

        return DataLoader(dataset, batch_size=self.args.batch_size, shuffle=(sampler is None),
                          num_workers=self.args.num_workers, sampler=sampler,
                          collate_fn=partial(EmotionDataset.collate_fn, pad_token_id=self.tokenizer.pad_token_id))

    def train_dataloader(self):
        self.train_dataloader_object = self._get_dataloader(self.train_dataloader_object, 'train', is_train=True)
        return self.train_dataloader_object

    def val_dataloader(self):
        self.val_dataloader_object = self._get_dataloader(self.val_dataloader_object, 'val', is_train=False)
        return self.val_dataloader_object

    def test_dataloader(self):
        self.test_dataloader_object = self._get_dataloader(self.test_dataloader_object, 'test', is_train=False)
        return self.test_dataloader_object

    def compute_weighted_loss(self, logits, labels):
        """ Weighted loss function """
        loss = F.cross_entropy(logits, labels, weight=self.loss_weights.to(labels.device), reduction='sum')
        loss /= labels.size(0)
        return loss

    def forward(self, input_ids, labels):
        attention_mask = self.get_attention_mask(input_ids)
        output = self.sentence_classifier_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return output

    def training_step(self, batch, batch_nb):
        output = self.forward(*batch)

        if self.args.balanced_weight_warming:
            loss = self.compute_weighted_loss(logits=output['logits'], labels=batch[1] )
        else:
            loss = output['loss'] # can be None if labels not set (predicting on test set)

        lr = loss.new_zeros(1) + self.trainer.optimizers[0].param_groups[0]['lr']
        tensorboard_logs = {'train_loss': loss, 'lr': lr,
                            'mem': torch.cuda.memory_allocated(loss.device) / 1024 ** 3 if torch.cuda.is_available() else 0}
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_nb):
        inputs, labels = batch
        outputs = self.forward(*batch)

        vloss = outputs['loss']
        scores = metrics.calculate_metrics(logits=outputs['logits'],
                                                        labels=labels,
                                                        emotions=self.train_set.get_emotions())
        scores['vloss'] = vloss
        return scores

    def validation_epoch_end(self, outputs):
        tqdm_dict = metrics.get_log_scores(outputs=outputs,
                                           emotions=self.train_set.get_emotions(),
                                           class_weights=self.class_weights)

        self.log('vloss', tqdm_dict["vloss"], prog_bar=False)
        self.log('valid_ac_unweighted', tqdm_dict["acc_unweighted"], prog_bar=False)
        self.log('macroF1', tqdm_dict["macroF1_score"], prog_bar=False)
        self.log('microF1', tqdm_dict["microF1_score"], prog_bar=False)

        microF1_per_class = {}
        for emotion_class in range(len(tqdm_dict["macroF1_per_class"])):
            microF1_per_class[str(emotion_class)] = tqdm_dict["macroF1_per_class"][emotion_class]
        self.log('F1 per class' ,microF1_per_class)

        if self.args.verbose:
            print(*tqdm_dict.items(), sep='\n')

        result = {'progress_bar': tqdm_dict, 'log': tqdm_dict, 'vloss': tqdm_dict["vloss"]}

        return result

    def test_step(self, batch, batch_nb):
        return self.validation_step(batch, batch_nb)

    def test_epoch_end(self, outputs):
        self.validation_epoch_end(outputs)
        #print(result)

    def configure_optimizers(self):
        """
        returns the optimizer and scheduler
        """
        params = self.layerwise_lr(self.args.lr, self.args.layerwise_decay)

        self.optimizer = torch.optim.Adam(params, lr=self.args.lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=10) # TODO: expose this as argument?
        return [self.optimizer], [self.scheduler]

    def layerwise_lr(self, lr, decay):
        """
        returns grouped model parameters with layer-wise decaying learning rate
        based on https://github.com/neeraj310/Master_Thesis_EA_In_ERC.git
        """
        m = self.sentence_classifier_model
        num_layers = m.config.num_hidden_layers
        opt_parameters = [{'params': m.bert.embeddings.parameters(), 'lr': lr*decay**num_layers}]
        opt_parameters += [{'params': m.bert.encoder.layer[l].parameters(), 'lr': lr*decay**(num_layers-l+1)}
                            for l in range(num_layers)]

        return opt_parameters

