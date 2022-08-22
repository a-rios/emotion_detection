# Emotion Detection from Text
This repository contains data and code for emotion detection in text, developped in the [EASIER project](https://www.project-easier.eu/) on sign language translation.
The end goal within the project is to provide emotional cues to an avatar for the translation direction from spoken to sign language. While the EASIER project is multilingual, emotion detection so far is restricted to German. 
Since datasets labeled with emotions are mostly limited to English, we use different methods of transfer to obtain a German version for two of the standard emotion recognition datasets:
 1. [MELD](https://affective-meld.github.io/): subtitles of the TV show Friends [3]
 2. [GoEmotions](https://github.com/monologg/GoEmotions-pytorch): Reddit posts [1]
 
## Transfer to German
### MELD
For the MELD Friends dataset, we use human-translated German subtitles obtained from the web and align them via timestamps and sentence similarity to the English subtitles (we measure sentence similarity with [LASER](https://github.com/facebookresearch/LASER)[4]). The alignment is somewhat noisy, as subtitles are not always split identically between the 2 languages, therefore, 1:n or n:1 alignments are possible.

### GoEmotions
For GoEmotions, there are no human translations available, instead we use machine translation to obtain the German version of the samples. The model we use for this purpose is [Facebook's winning WMT19 submission](https://huggingface.co/facebook/wmt19-en-de) for English->German translation [2].

## Emojis and Emoticons
GoEmotions has samples that contain emojis and/or emoticons. We use the [emot library](https://github.com/NeelShah18/emot) to restore emojis after translation. We also use emot to create additional versions of the GoEmotions data by replacing either only emojis or both emojis and emoticons with textual descriptions. We use the same model to translate the textual descriptions obtained from the emot library, see example below.

|| sample | label |
|----|--------|--------|
original:| This was so touching. :heart: :heart: | love |
replaced: | This was so touching. :red_heart: :red_heart: | love | 
translated: | Das war so ergreifend. :heart: :heart: | love |
replaced:| Das war so ergreifend. [rotes Herz] [rotes Herz] | love |

## References:
[1] Demszky, Dorottya, Dana Movshovitz-Attias, Jeongwoo Ko, Alan Cowen, Gaurav Nemade, and Sujith Ravi (2020). “[GoEmotions: A Dataset of Fine-Grained Emotions](https://aclanthology.org/2020.acl-main.372/)”. In: Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics. Online: Association for Computational Linguistics, pp. 4040–4054.

[2] Ng, Nathan, Kyra Yee, Alexei Baevski, Myle Ott, Michael Auli, and Sergey Edunov (2019). 
“[Facebook FAIR’s WMT19 News Translation Task Submission](https://aclanthology.org/W19-5333/)”. In: Proceedings of the Fourth Conference on Machine Translation (Volume 2: Shared Task Papers, Day 1). Florence, Italy: Association for Computational Linguistics, pp. 314–319.

[3] Poria, Soujanya, Devamanyu Hazarika, Navonil Majumder, Gautam Naik, Erik Cambria, and Rada Mihalcea (2019). “[MELD: A Multimodal Multi-Party Dataset for Emotion Recognition in Conversations](https://aclanthology.org/P19-1050/)”. In: Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics. Florence, Italy: Association for Computational Linguistics, pp. 527–536

[4] Schwenk, Holger and Matthijs Douze (2017). “[Learning Joint Multilingual Sentence Representations with Neural Machine Translation](https://arxiv.org/abs/1704.04154)”. In: Proceedings of the 2nd Workshop on Representation Learning for NLP. Vancouver, Canada: Association for Computational Linguistics, pp. 157–167. 
