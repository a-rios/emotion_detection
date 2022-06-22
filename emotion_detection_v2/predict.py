import os
import argparse
import numpy as np
import torch
import pytorch_lightning as pl
from . import data
from .model import EmotionPrediction
from .data import EmotionDataset

def main(args):
    model = EmotionPrediction.load_from_checkpoint(checkpoint_path=args.checkpoint)
    tokenizer =  model.get_tokenizer()

    test_set = EmotionDataset(in_file=args.test,
                                  tokenizer=tokenizer,
                                  file_format=args.file_format,
                                  label_name=args.label_name,
                                  utterance_name=args.utterance_name,
                                  split_name="test",
                                  emotions=model.emotions,
                                  remove_unaligned=not args.keep_unaligned,
                                  no_labels=False)
    model.set_testset(test_set)

    trainer = pl.Trainer(accelerator=args.accelerator, devices=args.devices, strategy='ddp_find_unused_parameters_false' if torch.cuda.is_available() else None)
    trainer.test(model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Emotion Prediction")

    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument("--accelerator", type=str, default="gpu", help="Pytorch lightning accelerator argument: cpu or gpu. Default: gpu.")
    parser.add_argument("--devices", type=int, nargs="+", required=True, help="Device id(s).")
    parser.add_argument("--cache_dir", type=str, default=None, help="Cache directory for huggingface models.")
    parser.add_argument("--test", type=str, default=None, required=True, help="Path to the source test file (to evaluate after training is finished).")
    parser.add_argument("--file_format", type=str, default="json", help="Input format, options are: json, csv, text (for prediction only). Default: json.")
    parser.add_argument("--label_name", type=str, help="Key/column name for labels.")
    parser.add_argument("--utterance_name", type=str, help="Key/column name for utterances.")
   # parser.add_argument("--prediction_only", action="store_true", help="Test set for prediction only, no labels, do not calculate scores.")
    parser.add_argument('--checkpoint', type=str, metavar='PATH' ,required=True, help='Path to checkpoint of trained model.')
    parser.add_argument('--json_out', type=str, metavar='PATH',  help='Path to json output file.')
    parser.add_argument('--csv_out', type=str, metavar='PATH',  help='Path to csv output file.')
    parser.add_argument("--keep_unaligned", action='store_true', help="Keep samples where sting == 'NOT FOUND' (aligned German Friends set).")
    parser.add_argument("--no_labels", action='store_true', help="Test set has no labels (skip calculating metrics).")
    #parser.add_argument('--print_logits',  action="store_true", help = 'Print raw logits to json instead of probabilities.')
    args = parser.parse_args()
    main(args)

## TODO: predict without labels, print logits to csv/json
