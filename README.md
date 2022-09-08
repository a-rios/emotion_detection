# Emotion Detection from Text
This repository contains data and code for emotion detection in text, developped in the [EASIER project](https://www.project-easier.eu/) on sign language translation.
The end goal within the project is to provide emotional cues to an avatar for the translation direction from spoken to sign language. While the EASIER project is multilingual, emotion detection so far is restricted to German. 

## Data
Since datasets labeled with emotions are mostly limited to English, we use different methods of transfer to obtain a German version for two of the standard emotion recognition datasets:
 1. [MELD](https://affective-meld.github.io/): subtitles of the TV show Friends [3]
 2. [GoEmotions](https://github.com/monologg/GoEmotions-pytorch): Reddit posts [1]
 
### Transfer to German
#### MELD
For the MELD Friends dataset, we use human-translated German subtitles obtained from the web and align them via timestamps and sentence similarity to the English subtitles (we measure sentence similarity with [LASER](https://github.com/facebookresearch/LASER)[4]). The alignment is somewhat noisy, as subtitles are not always split identically between the 2 languages, therefore, 1:n or n:1 alignments are possible.

#### GoEmotions
For GoEmotions, there are no human translations available, instead we use machine translation to obtain the German version of the samples. The model we use for this purpose is [Facebook's winning WMT19 submission](https://huggingface.co/facebook/wmt19-en-de) for English->German translation [2].

##### Emojis and Emoticons
GoEmotions has samples that contain emojis and/or emoticons. We use the [emot library](https://github.com/NeelShah18/emot) to restore emojis after translation. We also use emot to create additional versions of the GoEmotions data by replacing either only emojis or both emojis and emoticons with textual descriptions. We use the same model to translate the textual descriptions obtained from the emot library, see example below.

|| sample | label |
|----|--------|--------|
original:| This was so touching. :heart: :heart: | love |
replaced: | This was so touching. :red_heart: :red_heart: | love | 
translated: | Das war so ergreifend. :heart: :heart: | love |
replaced:| Das war so ergreifend. [rotes Herz] [rotes Herz] | love |

## Installation
Checkout and install the repository:

```
git clone https://github.com/a-rios/emotion_detection.git
cd emotion_detection
pip install -e .
```

## Fine-Tuning
The code loads and train an instance of huggingface [Automodel](https://huggingface.co/docs/transformers/model_doc/auto) with a classification head, in principle any of those models *should* be usable - however, only BERT and gpt2 models have been tested with the code, other models might need some adaptions.

Some important settings:
* `--file_format`: either csv or json
* `--utterance_name`: column name/key that contains the text samples for the model to encode
* `--label_name`: column name/key that contains the labels
* `--num_classes`: number of classes in the dataset
* `--early_stopping_metric`: which validation metric to use for early stopping (`vloss`, `valid_ac_unweighted`, `macroF1`, `microF1`)
* `--keep_unaligned`: Friends dataset only, samples that have no German translation because no alignment was found have a placeholder (`NOT FOUND`), by default, those are skipped for training. Use this option to keep them.

Example for fine-tuning German BERT on the Friends dataset:

```
python -m emotion_detection.main \
--tokenizer "dbmdz/bert-base-german-uncased" \
--from_pretrained "dbmdz/bert-base-german-uncased" \
--accelerator gpu \
--devices device-id \
--batch_size 32 \
--val_every 1.0 \
--cache_dir path-to-huggingface-cache-dir \
--train data/MELD_Friends_German/train_de.csv \
--dev data/MELD_Friends_German/dev_de.csv \
--test data/MELD_Friends_German/test_de.csv \
--early_stopping_metric microF1 \
--file_format csv \
--utterance_name utterance_de \
--label_name label \
--save_dir path-to-save-the-model \
--save_prefix name-of-model \
--seed random-seed \
--save_top_k 1 \
--num_classes 7 \
--scheduler 'cosine' \
--dropout 0.1 \
--classifier_dropout 0.1 \
--patience 10 \
--verbose \
--keep_unaligned \
--fp32
```

Example for fine-tuning on GoEmotions:
```
python -m emotion_detection.main \
--tokenizer "dbmdz/bert-base-german-uncased" \
--from_pretrained "dbmdz/bert-base-german-uncased" \
--accelerator gpu \
--devices $device \
--batch_size 8 \
--grad_accum 4 \
--val_every 1.0 \
--cache_dir $cache_dir \
--train data/GoEmotions_German/train.csv \
--dev data/GoEmotions_German/dev.csv \
--test data/GoEmotions_German/test.csv \
--early_stopping_metric microF1 \
--file_format csv \
--utterance_name de_text \
--label_name original_label \
--save_dir path-to-save-the-model \
--save_prefix model-name \
--seed random-seed \
--save_top_k 1 \
--num_classes 28 \
--scheduler 'cosine' \
--dropout 0.1 \
--classifier_dropout 0.1 \
--patience 10 \
--verbose \
--fp32
```

See [main.py](emotion_detection/main.py) for a full list of options.

## Prediction

Predicting emotions with a trained model can be done as follows:

```
python -m emotion_detection.predict \
--checkpoint path-to-fine-tuned-checkpoint \
--accelerator gpu \
--devices device-id \
--no_progress_bar \
--batch_size 32 \
--input_format csv \
--utterance_name your-text-name \
--label_name your-label-name \
--test your-test.csv
```

Some extra options for predictions:
* `--column_names`: if your csv does not have a header, specify the names here (e.g. `"A,B,C"`)
* `--csv_delimiter`: set the delimiter in your csv (default is ",")
* `--no_labels`: test set has no labels, only predict, do not score
* `--output_format`: csv or json, will print raw probabilites (logits) for each sample.
* if the test set contains labels (and `--no_labels` is not set ) and no output format is given, will run the prediction and print the scores to stdout


## References:
[1] Demszky, Dorottya, Dana Movshovitz-Attias, Jeongwoo Ko, Alan Cowen, Gaurav Nemade, and Sujith Ravi (2020). “[GoEmotions: A Dataset of Fine-Grained Emotions](https://aclanthology.org/2020.acl-main.372/)”. In: Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics. Online: Association for Computational Linguistics, pp. 4040–4054.

[2] Ng, Nathan, Kyra Yee, Alexei Baevski, Myle Ott, Michael Auli, and Sergey Edunov (2019). 
“[Facebook FAIR’s WMT19 News Translation Task Submission](https://aclanthology.org/W19-5333/)”. In: Proceedings of the Fourth Conference on Machine Translation (Volume 2: Shared Task Papers, Day 1). Florence, Italy: Association for Computational Linguistics, pp. 314–319.

[3] Poria, Soujanya, Devamanyu Hazarika, Navonil Majumder, Gautam Naik, Erik Cambria, and Rada Mihalcea (2019). “[MELD: A Multimodal Multi-Party Dataset for Emotion Recognition in Conversations](https://aclanthology.org/P19-1050/)”. In: Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics. Florence, Italy: Association for Computational Linguistics, pp. 527–536

[4] Schwenk, Holger and Matthijs Douze (2017). “[Learning Joint Multilingual Sentence Representations with Neural Machine Translation](https://arxiv.org/abs/1704.04154)”. In: Proceedings of the 2nd Workshop on Representation Learning for NLP. Vancouver, Canada: Association for Computational Linguistics, pp. 157–167. 
