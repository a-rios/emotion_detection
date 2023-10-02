import os
import argparse
import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ProgressBar
import os
from . import data
from .model import EmotionPrediction
from .data import EmotionDataset

class LitProgressBar(ProgressBar):
        def init_test_tqdm(self):
            bar = super().init_test_tqdm()
            bar.set_description('predicting..')
            bar.disable = True
            return bar

def main(args):
    model = EmotionPrediction.load_from_checkpoint(checkpoint_path=args.checkpoint, map_location="cpu", cache_dir=args.cache_dir)
    tokenizer =  model.get_tokenizer()

    save_texts = True if args.output_format is not None else False # if we need to print results per sample, store original text with id in data set (so we don't have to reconstruct this from the tokenizer)

    test_set = EmotionDataset(in_file=args.test,
                                  tokenizer=tokenizer,
                                  file_format=args.input_format,
                                  label_name=args.label_name,
                                  utterance_name=args.utterance_name,
                                  split_name="test",
                                  max_length=model.max_input_length,
                                  emotions=model.emotions,
                                  remove_unaligned=not args.keep_unaligned,
                                  no_labels=args.no_labels,
                                  save_texts=save_texts,
                                  csv_delimiter=args.csv_delimiter,
                                  csv_names=args.column_names)
    model.set_testset(test_set, out_format=args.output_format, out_file=args.output_file, print_logits=args.print_logits)
    if args.no_labels:
        model.set_no_labels_run()

    if args.no_progress_bar:
        progress_bar_callback = LitProgressBar()
    else:
        progress_bar_callback = ProgressBar(refresh_rate=args.progress_bar_refresh_rate)

    trainer = pl.Trainer(accelerator="ddp_cpu",
                         callbacks=[progress_bar_callback])
    trainer.test(model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Emotion Prediction")

    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument("--test", type=str, default=None, metavar='PATH', required=True, help="Path to the test file for evaluation.")
    parser.add_argument("--input_format", type=str, default="json", required=True, help="Input format, options are: json, csv. Default: json.")
    parser.add_argument("--csv_delimiter", type=str, default=",", help="Delimiter to read in csv. Default: comma.")
    parser.add_argument("--label_name", type=str, help="Key/column name for labels.")
    parser.add_argument("--utterance_name", type=str, required=True, help="Key/column name for utterances.")
    parser.add_argument('--checkpoint', type=str, required=True, metavar='PATH', help='Path to checkpoint of trained model.')
    parser.add_argument('--output_format', type=str, default=None, help='Output format (csv, json or None). If None, will print results to stdout (without logits/probs).')
    parser.add_argument("--print_logits", action='store_true', help="Print logits instead of probabilities.")
    parser.add_argument('--output_file', type=str, metavar='PATH',  help='Path to output file (json or csv).')
    parser.add_argument("--keep_unaligned", action='store_true', help="Keep samples where sting == 'NOT FOUND' (aligned German Friends set).")
    parser.add_argument("--no_labels", action='store_true', help="Test set has no labels (skip calculating metrics).")
    parser.add_argument("--no_progress_bar", action='store_true', help="Disable progress bar printing.")
    parser.add_argument("--column_names", type=str, help="If csv does not include column names, provide them as a comma separated list of strings ('A,B,C').")
    parser.add_argument("--cache_dir", type=str, default=None, help="Cache directory for huggingface models.")
    args = parser.parse_args()
    main(args)

