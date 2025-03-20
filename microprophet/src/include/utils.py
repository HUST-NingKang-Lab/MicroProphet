import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from include.evaluator import Evaluator
from include.MicroCorpus import SequenceClassificationDataset
import os

def seed_everything(seed: int):
    import random, os
    import numpy as np
    import torch
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def eval_and_save(y_score, y_true, label_names, save_dir, activation="softmax"):
    if activation == "sigmoid":
        y_score = nn.Sigmoid()(torch.tensor(y_score)).numpy()
    elif activation == "softmax":
        y_score = nn.Softmax(dim=1)(torch.tensor(y_score)).numpy()
    elif activation == "none":
        y_score = torch.tensor(y_score).numpy()
    # y_score = nn.Softmax(dim=1)(torch.tensor(y_score)).numpy()
    evaluator = Evaluator(y_score, y_true, label_names=label_names, num_thresholds=100)
    metrics, avg_metrics = evaluator.eval()

    # save
    for label, metric in metrics.items():
        os.makedirs(os.path.join(save_dir), exist_ok=True)
        metric.to_csv(os.path.join(save_dir, f"{label}.csv"))
    avg_metrics.to_csv(os.path.join(save_dir, "avg.csv"))
    return avg_metrics


# ----------------------------- generation utils -----------------------------
def clfset_from_sent(sent, le, tokenizer):
        labels = sent[:, 1]
        labels = [tokenizer.decode([label]) for label in labels]
        from sklearn.preprocessing import LabelEncoder, OneHotEncoder
        if isinstance(le, OneHotEncoder):
            labels = le.transform(np.array(labels).reshape(-1, 1)).toarray().argmax(1)
        elif isinstance(le, LabelEncoder):
            labels = le.transform(np.array(labels).reshape(-1, 1))
        else:
            raise ValueError("le must be either LabelEncoder or OneHotEncoder")
        tokens = torch.cat([sent[:, 0].view(-1,1), 
                                sent[:, 2:], 
                                torch.zeros(sent.shape[0], 1, dtype=torch.long)], # pad to 512
                               dim=1)
        tokens[tokens >= 9669] = 0
        attn_mask = torch.ones_like(tokens)
        attn_mask[tokens == tokenizer.pad_token_id] = 0
        set = SequenceClassificationDataset(
            seq=tokens,
            mask=attn_mask,
            labels=labels)
        return set

def clfset_from_labeled_corpus(corpus, le, tokenizer):
        labels = [tokenizer.decode(label) for label in corpus.labels]
        from sklearn.preprocessing import LabelEncoder, OneHotEncoder
        if isinstance(le, OneHotEncoder):
            labels = le.transform(np.array(labels).reshape(-1, 1)).toarray()
        elif isinstance(le, LabelEncoder):
            labels = le.transform(np.array(labels).reshape(-1, 1))
        else:
            raise ValueError("le must be either LabelEncoder or OneHotEncoder")
        labels = labels.argmax(axis=1)
        tokens = corpus[:]['input_ids']
        tokens = torch.cat([tokens[:, 0].view(-1,1),
                            tokens[:, 2:],
                            torch.zeros(tokens.shape[0], 1, dtype=torch.long)], # pad to 512
                           dim=1)
        mask = corpus[:]["attention_mask"][:,1:]
        mask = torch.cat([mask, torch.zeros(mask.shape[0], 1, dtype=torch.long)], dim=1)
        clf_set = SequenceClassificationDataset(
            seq=tokens,
            mask=mask,
            labels=labels)
        return clf_set