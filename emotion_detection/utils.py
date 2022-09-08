import json
from . import data
from typing import List
import re

def _get_idx(sample: str):
        m = re.search('^(\d+):\s*(.*)', sample)
        if m:
            idx = m.group(1)
            line = m.group(2)
            return idx, line
        else:
            print(f"No idx found in line {sample}.")
            exit(1)

def probs_to_json(outputs: List[dict],
                  test_set: data.EmotionDataset,
                  out_file: str,
                  out_format: str,
                  emotions_inv: dict):
    result = []
    for output in outputs:  # batched
        batch_logits = output['logits'].cpu()
        for i, text in enumerate(output['texts']):
            idx, text = _get_idx(text)
            sample_logits = batch_logits[i,:].tolist()
            emotion_dict = {emotions_inv[n]:logit  for n, logit in enumerate(sample_logits) }
            sample_dict = {"id": idx,
                           "text": text,
                           "lang": "de_DE",
                           "gender": "unknown", #TODO: not available with all datasets
                           "emotion-text": emotion_dict}
            result.append(sample_dict)

    if out_format == 'json':
        json_data = json.dumps(result, indent=3, ensure_ascii=False)
        with open(out_file, 'w') as f:
            f.write(json_data)
    elif out_format == 'csv':
        with open(out_file, 'w') as f:
            f.write("id,text,lang,gender,")
            for k, e in emotions_inv.items():
                if k == len(emotions_inv)-1:
                    f.write(f"{e}\n")
                else:
                    f.write(f"{e},")
            for s in result:
                f.write(f"{s['id']},{s['text']},{s['lang']},{s['gender']},")
                for i,(k,v) in enumerate(s['emotion-text'].items()):
                    if i == len(s['emotion-text'])-1:
                        f.write(f"{v}\n")
                    else:
                        f.write(f"{v},")
    else:
        print(f"Unsupported output format {out_format}. Valid options are 'json' or 'csv'.")
