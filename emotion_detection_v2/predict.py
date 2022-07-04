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

    save_texts = True if args.output_format is not None else False # if we need to print results per sample, store original text with id in data set (so we don't have to reconstruct this from the tokenizer)

    test_set = EmotionDataset(in_file=args.test,
                                  tokenizer=tokenizer,
                                  file_format=args.input_format,
                                  label_name=args.label_name,
                                  utterance_name=args.utterance_name,
                                  split_name="test",
                                  emotions=model.emotions,
                                  remove_unaligned=not args.keep_unaligned,
                                  no_labels=args.no_labels,
                                  save_texts=save_texts,
                                  csv_delimiter=args.csv_delimiter)
    model.set_testset(test_set, out_format=args.output_format, out_file=args.output_file)
    if args.no_labels:
        model.set_no_labels_run()

    trainer = pl.Trainer(accelerator=args.accelerator, devices=args.devices, strategy='ddp_find_unused_parameters_false' if torch.cuda.is_available() else None)
    trainer.test(model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Emotion Prediction")

    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument("--accelerator", type=str, default="gpu", help="Pytorch lightning accelerator argument: cpu or gpu. Default: gpu.")
    parser.add_argument("--devices", type=int, nargs="+", required=True, help="Device id(s).")
    #parser.add_argument("--cache_dir", type=str, default=None, help="Cache directory for huggingface models.")
    parser.add_argument("--test", type=str, default=None, required=True, help="Path to the source test file (to evaluate after training is finished).")
    parser.add_argument("--input_format", type=str, default="json", required=True, help="Input format, options are: json, csv. Default: json.")
    parser.add_argument("--csv_delimiter", type=str, default=",", help="Delimiter to read in csv. Default: comma.")
    parser.add_argument("--label_name", type=str, help="Key/column name for labels.")
    parser.add_argument("--utterance_name", type=str, required=True, help="Key/column name for utterances.")
    parser.add_argument('--checkpoint', type=str, required=True, metavar='PATH', help='Path to checkpoint of trained model.')
    parser.add_argument('--output_format', type=str, default=None, help='Output format (csv, json or None). If None, will print results to stdout (without logits).')
    parser.add_argument('--output_file', type=str, metavar='PATH',  help='Path to output file (json or csv).')
    parser.add_argument("--keep_unaligned", action='store_true', help="Keep samples where sting == 'NOT FOUND' (aligned German Friends set).")
    parser.add_argument("--no_labels", action='store_true', help="Test set has no labels (skip calculating metrics).")
    args = parser.parse_args()
    main(args)

## TODO: predict without labels
