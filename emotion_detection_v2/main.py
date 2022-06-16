import argparse
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import TQDMProgressBar
from pytorch_lightning.plugins import DDPPlugin
import random
import logging
import numpy as np
import os
from transformers import AutoModel, AutoTokenizer, AutoConfig
from .data import EmotionDataset
from .model import EmotionPrediction

logger = logging.getLogger('pytorch_lightning')
logging.basicConfig(level=logging.INFO)

def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)


    if args.wandb:
        logger = WandbLogger(project=args.wandb, entity=args.wandb_entity)
    else:
        logger = TensorBoardLogger(save_dir=os.path.join(args.save_dir, args.save_prefix), name="tensorboard_logs")

    model = EmotionPrediction(args)
    tokenizer =  model.get_tokenizer()

    train_set = EmotionDataset(in_file=args.train,
                                  tokenizer=tokenizer,
                                  file_format=args.file_format,
                                  label_name=args.label_name,
                                  utterance_name=args.utterance_name,
                                  split_name="train",
                                  remove_unaligned=not args.keep_unaligned,
                                  prediction_only=False)
    emotions = train_set.get_emotions()
    max_len = train_set.get_max_len()
    dev_set = EmotionDataset(in_file=args.dev,
                                  tokenizer=tokenizer,
                                  file_format=args.file_format,
                                  label_name=args.label_name,
                                  utterance_name=args.utterance_name,
                                  split_name="dev",
                                  remove_unaligned=not args.keep_unaligned,
                                  emotions=emotions,
                                  max_len=max_len,
                                  prediction_only=False)
    if args.test:
        test_set = EmotionDataset(in_file=args.test,
                                    tokenizer=tokenizer,
                                    file_format=args.file_format,
                                    label_name=args.label_name,
                                    utterance_name=args.utterance_name,
                                    split_name="test",
                                    remove_unaligned=not args.keep_unaligned,
                                    emotions=emotions,
                                    max_len=max_len,
                                    prediction_only=False) # TODO predition test set (no labels)

    model.set_datasets(train_set=train_set,
                            dev_set=dev_set,
                            test_set=test_set)

    model.lr_mode='min' if args.early_stopping_metric == 'vloss' else 'max'
    early_stop_callback = EarlyStopping(monitor=args.early_stopping_metric, min_delta=args.min_delta, patience=args.patience, verbose=True, mode=model.lr_mode)
    progress_bar_callback = TQDMProgressBar(refresh_rate=args.progress_bar_refresh_rate)

    checkpoint_name = "checkpoint{{epoch:02d}}_{{{}".format(args.early_stopping_metric)
    checkpoint_name += ':.3f}'

    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(args.save_dir, args.save_prefix),
        filename=checkpoint_name,
        save_top_k=args.save_top_k,
        verbose=True,
        monitor=args.early_stopping_metric,
        mode=model.lr_mode)

    trainer = pl.Trainer(accelerator=args.accelerator, devices=args.devices, strategy='ddp_find_unused_parameters_false' if torch.cuda.is_available() else None,
                         track_grad_norm=-1,
                         max_steps=-1 if not args.debug else 1,
                         replace_sampler_ddp=False,
                         accumulate_grad_batches=args.grad_accum,
                         val_check_interval=args.val_every if not args.debug else 1,
                         num_sanity_val_steps=args.num_sanity_val_steps,
                         check_val_every_n_epoch=1 if not (args.debug) else 1,
                         limit_val_batches=args.val_percent_check,
                         limit_test_batches=False,
                         logger=logger,
                         enable_checkpointing=True if not args.disable_checkpointing else False,
                         precision=32 if args.fp32 else 16, amp_backend='native', # amp_backend='apex', amp_level='O2', -> gradient overflows, can't use it
                         resume_from_checkpoint=args.resume_ckpt,
                         callbacks=[early_stop_callback, checkpoint_callback, progress_bar_callback]
                         )
    ## write config + tokenizer to save_dir
    model.sentence_classifier_model.save_pretrained(args.save_dir + "/" + args.save_prefix)
    model.tokenizer.save_pretrained(args.save_dir + "/" + args.save_prefix)
    trainer.fit(model)
    print("Training ended. Best checkpoint {} with {} {}.".format(model.best_checkpoint, model.best_metric, args.early_stopping_metric))
    trainer.test(model)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Emotion Detection")
    # pretrained args
    parser.add_argument("--tokenizer", type=str, help="Path to the tokenizer directory.")
    parser.add_argument("--from_pretrained", type=str, default=None,  help="Path to a checkpoint to load model weights but not training state")
    parser.add_argument("--save_dir", type=str, default='simplification', help="Directory to save models.")
    parser.add_argument("--save_prefix", type=str, default='test', help="subfolder in save_dir for this model")
    parser.add_argument("--resume_ckpt", type=str, help="Path of a checkpoint to resume from")
    parser.add_argument("--num_sanity_val_steps", type=int, default=0,  help="Number of evaluation sanity steps to run before starting the training. Default: 0.")
    parser.add_argument("--cache_dir", type=str, default=None, help="Cache directory for huggingface models.")

    # data args
    parser.add_argument("--train", type=str, default=None,  help="Path to the source train file.")
    parser.add_argument("--dev", type=str, default=None, help="Path to the source validation file.")
    parser.add_argument("--test", type=str, default=None, help="Path to the source test file (to evaluate after training is finished).")
    parser.add_argument("--file_format", type=str, default="json", help="Input format, options are: json, csv, text (for prediction only). Default: json.")
    parser.add_argument("--label_name", type=str, help="Key/column name for labels.")
    parser.add_argument("--utterance_name", type=str, help="Key/column name for utterances.")
    parser.add_argument("--keep_unaligned", action='store_true', help="Keep samples where sting == 'NOT FOUND' (aligned German Friends set).")

    # model args
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of data loader workers")
    parser.add_argument("--grad_accum", type=int, default=1, help="Number of gradient accumulation steps.")
    parser.add_argument("--accelerator", type=str, default="gpu", help="Pytorch lightning accelerator argument: cpu or gpu. Default: gpu.")
    parser.add_argument("--devices", type=int, nargs="+", required=True, help="Device id(s).")
    parser.add_argument("--seed", type=int, default=42, help="Seed")
    parser.add_argument("--attention_dropout", type=float, default=0.1, help="attention dropout")
    parser.add_argument("--dropout", type=float, default=0.1, help="dropout")
    parser.add_argument("--activation_dropout", type=float, default=0.0, help="activation_dropout")
    parser.add_argument('--layerwise_decay', default=0.95, type=float, help='layerwise decay factor for the learning rate of the pretrained Bert models.')
    parser.add_argument('--balanced_weight_warming', action="store_true", help = 'Use balanced weight warming for loss function')
    parser.add_argument("--weight_rate", type=float, default=1.0, help="Weight rate for scaling losses w.r.t. class frequency.")
    parser.add_argument("--num_classes", type=int, help="Number of emotion categories to learn.")


    # optimization args:
    parser.add_argument("--lr", type=float, default=0.00003, help="Initial learning rate")
    parser.add_argument("--val_every", type=float, default=1.0, help="Number of training steps between validations in percent of an epoch.")
    parser.add_argument("--val_percent_check", default=1.00, type=float, help='Percent of validation data used')
    parser.add_argument("--max_epochs", type=int, default=100000, help="Maximum number of epochs (will stop training even if patience for early stopping has not been reached).")
    parser.add_argument("--early_stopping_metric", type=str, default='vloss', help="Metric to be used for early stopping: vloss, valid_ac_unweighted, macroF1, microF1")
    parser.add_argument("--patience", type=int, default=10, help="Patience for early stopping.")
    parser.add_argument("--min_delta", type=float, default=0.0, help="Minimum change in the monitored quantity to qualify as an improvement.")
    parser.add_argument("--lr_reduce_patience", type=int, default=8, help="Patience for LR reduction in Plateau scheduler.")
    parser.add_argument("--lr_reduce_factor", type=float, default=0.5, help="Learning rate reduce factor for Plateau scheduler.")
    parser.add_argument("--disable_checkpointing", action='store_true', help="No logging or checkpointing")
    parser.add_argument("--save_top_k", type=int, default=5, help="Number of best checkpoints to keep. Others will be removed.")
    parser.add_argument('--grad_ckpt', action='store_true', help='Enable gradient checkpointing to save memory')

    # logging args
    parser.add_argument("--progress_bar_refresh_rate", type=int, default=0, help="How often to refresh progress bar (in steps). Value 0 disables progress bar.")
    parser.add_argument("--fp32", action='store_true', help="default is fp16. Use --fp32 to switch to fp32")
    parser.add_argument("--wandb", type=str, default=None, help="WandB project name to use if logging fine-tuning with WandB.")
    parser.add_argument("--wandb_entity", type=str, default=None, help="WandB entity name to use if logging fine-tuning with WandB.")
    parser.add_argument("--debug", action='store_true', help="Debugging run (1 step only).")
    parser.add_argument("--verbose", action='store_true', help="Print validation results to stdout .")
    #parser.add_argument("--print_params", action='store_true', help="Print parameter names and shapes.")


    args = parser.parse_args()
    main(args)
