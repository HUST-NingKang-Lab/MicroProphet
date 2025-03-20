import torch
from pickle import dump, load
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
from typing import Optional, List, Dict, Union, Tuple

class MicroTokenizer(PreTrainedTokenizer):
    def __init__(self, toks, **kwargs):
        super(MicroTokenizer, self).__init__(**kwargs)
        self.toks = []
        self.add_special_tokens({'pad_token': '<pad>', 
                                 'mask_token': '<mask>',
                                 'cls_token': '<cls>'})
        self.toks = self.toks + toks
        self.vocab = {v: i for i, v in enumerate(self.toks)}
        self.ids_to_tokens = {i: v for i, v in enumerate(self.toks)}
    
    def _tokenize(self, text):
        return list(text)
    
    def _add_tokens(self, new_tokens: List[str], special_tokens: bool = False) -> int:
        self.toks.extend(new_tokens)
        self.vocab = {v: i for i, v in enumerate(self.toks)}
        self.ids_to_tokens = {i: v for i, v in enumerate(self.toks)}
        return len(new_tokens)

    
    def _convert_token_to_id(self, token):
        return self.vocab[token]
    
    def _convert_id_to_token(self, index):
        return self.ids_to_tokens[index]
    
    def get_vocab(self):
        return self.vocab
    
    def get_vocab_size(self):
        return len(self.vocab)
    
    @property
    def vocab_size(self):
        return len(self.vocab)

class MicroCorpus(Dataset):
    def __init__(self, 
                 tokenizer: PreTrainedTokenizer,
                 data_path: Optional[str]=None,
                 abu: Optional[pd.DataFrame]=None,
                 key='genus',
                 max_len=512,
                 is_binning=True,
                 bins=51,
                 lm=False,  # language modeling
                 mlm_rate=None # None for causal language model
                 ):
        
        if data_path:
            file_type = data_path.split('.')[-1]
            if file_type not in ['h5', 'csv', 'tsv', 'txt']:
                raise ValueError('File type not supported.'
                                 'Please provide h5, csv, tsv or txt file.')
            if file_type == 'h5':
                self.data = pd.read_hdf(data_path, key=key).T
            else:
                sep = ',' if file_type == 'csv' else '\t'
                self.data = pd.read_csv(data_path, sep=sep, index_col=0).T
        elif abu is not None:
            self.data = abu
        else:
            raise ValueError('Please provide data_path or abu.')
        self.tokenizer = tokenizer
        self.max_len = max_len
        
        # keep only genus level and binning relative abundance
        self.data = self._preprocess(self.data)
        self.is_binning = is_binning
        self.bins = bins
        self.lm = lm
        self.mlm_rate = mlm_rate
    
        # convert to token
        tokens_list = []
        values_list = []
        length_list = []
        
        for sample in tqdm(self.data.index):
            tokens, value, length = self._convert_to_token(self.data.loc[sample], is_binning)
            tokens_list.append(tokens)
            values_list.append(value)
            length_list.append(length)
            
        # del self.data   # for saving memory
        print(f'Total {len(tokens_list)} samples.\n\
            Max length is {max(length_list)}.\n\
            Average length is {np.mean(length_list)}.\n\
            Min length is {min(length_list)}.')
        self.tokens = torch.LongTensor(tokens_list)
        self.values = torch.tensor(values_list)
    
    def __getitem__(self, index):
        attention_mask = torch.ones(self.tokens[index].shape)
        attention_mask[self.tokens[index] == self.tokenizer.pad_token_id] = 0
        tokens = self.tokens[index].clone()
        values = self.values[index].clone()
        
        if self.lm and self.mlm_rate:   # masked language model
            # masking
            labels = values.clone()
            mask = torch.full(labels.shape, self.mlm_rate)
            mask = mask * attention_mask    # only mask non-padding tokens
            
            if len(tokens.shape) == 1:  # handle single sample
                mask[:1] = 0  # ignore <cls>
            else:
                mask[:, :1] = 0  # ignore <cls>
                
            mask = torch.bernoulli(mask).bool()
            labels[~mask] = -100
            
            # 80% of the time, we replace masked input tokens with masking token
            indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & mask
            values[indices_replaced] = self.bins
            

            # 10% of the time, we replace masked input tokens with random word
            indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & mask & ~indices_replaced
            if self.is_binning:
                random_words = torch.randint(self.bins, labels.shape, dtype=torch.long)
                values[indices_random] = random_words[indices_random]
            else:   # if not binning, replace with random relative abundance
                random_words = torch.rand(labels.shape, dtype=torch.float)
                values[indices_random] = random_words[indices_random]
                
            # The rest of the time (10% of the time) we keep the masked input tokens unchanged
            
            return {'input_ids': torch.tensor(tokens),
                    'values': values.type(torch.float),
                    'attention_mask': attention_mask,
                    'labels': labels}
            
        elif self.lm and not self.mlm_rate:   # causal language model
            labels = values.clone()
            labels[attention_mask == 0] = -100
                
            return {'input_ids': torch.tensor(tokens),
                    'values': values.type(torch.float),
                    'attention_mask': attention_mask,
                    'labels': labels}

        return {'input_ids': torch.tensor(tokens),
                'values': torch.tensor(values, dtype=torch.float),
                'attention_mask': attention_mask}
    
    def __len__(self):
        return len(self.tokens)        
        
    def _convert_to_token(self, sample, is_binning):
        # drop zero values
        sample = sample[sample != 0]
        # -log1p
        if is_binning:
            # sample = np.log1p(sample)
            sample = pd.cut(sample, bins=self.bins, labels=False)
        # else:
        #     sample = np.log1p(sample)
        sent = sample.index.tolist()
        value = sample.values.tolist()
        
        # add cls
        sent = ['<cls>'] + sent
        value = [0] + value

        # convert to token
        tokens = self.tokenizer.encode(sent)
        length = len(tokens)
        
        # padding and truncate
        if len(tokens) > self.max_len:
            tokens = tokens[:self.max_len]
            value = value[:self.max_len]
            
        else:
            tokens.extend([self.tokenizer.pad_token_id] * (self.max_len - len(tokens)))
            value.extend([0] * (self.max_len - len(value)))
            
        return tokens, value, length
    
    def _preprocess(self, data):
        # data.columns = data.columns.str.replace('; ', ';', regex=False) # remove space after ;
        # data.columns = data.columns.str.replace(';s__.*', '', regex=True) # drop species level
        # data.columns = data.columns.str.replace('^k__', 'sk__', regex=True) # if start with k__, replace with sk__
        # extract 'g__XXX' in the column names
        data.columns = data.columns.str.extract(r'(g__[A-Za-z0-9_]+)').squeeze()
        data = data.groupby(data.columns, axis=1).sum()
        before = data.shape[0]
        # only keep genus in token list
        target_df = pd.DataFrame(index=self.tokenizer.toks)
        data = target_df.merge(data.T, left_index=True, right_index=True, how='left').fillna(0).T
        # drop all zero rows
        data = data.loc[(data != 0).any(axis=1)]
        print(f'{before - data.shape[0]} samples are dropped for all zeroes')
        # relative abundance
        data = data.div(data.sum(axis=1), axis=0)
        return data
    
class SequenceClassificationDataset(Dataset):
    def __init__(self, seq, value, mask, labels):
        self.seq = seq
        self.value = value
        self.mask = mask
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(self.seq[idx]),
            "values": torch.tensor(self.value[idx]),
            "attention_mask": torch.tensor(self.mask[idx]),
            "labels": torch.tensor(self.labels[idx])
        }
        
class MicroCorpusWithLabelTokens(Dataset):
    def __init__(self, tokens, labels, tokenizer):
        self.tokens = tokens
        self.tokenizer = tokenizer
        self.labels = torch.tensor(self.tokenizer.encode(labels)).view(-1, 1)
        # insert label tokens after <bos> 
        self.tokens = torch.cat((self.tokens[:, :1], self.labels, self.tokens[:, 1:-1]), dim=1)
        
    def __len__(self):
        return self.tokens.shape[0]
    
    def __getitem__(self, idx):
        attention_mask = torch.ones(self.tokens[idx].shape)
        attention_mask[self.tokens[idx] == self.tokenizer.pad_token_id] = 0
        tokens = self.tokens[idx].clone()

        return {'input_ids': torch.tensor(tokens),
                'attention_mask': attention_mask}
        
if __name__ == '__main__':
    # create MicroCorpus using MGnify data
    # abu = pd.read_hdf('data/abu_processed.h5', 'genus')
    abu = pd.read_hdf('~/data4/projects/microformer/data/abu.h5', 'genus')
    phylogeny = pd.read_csv('data/phylogeny.csv', index_col=0)
    genus_toks = phylogeny.index.tolist()
    tokenizer = MicroTokenizer(genus_toks)
    
    dump(tokenizer, open('MicroTokenizer.pkl', 'wb'))
    
    corpus = MicroCorpus(abu=abu, tokenizer=tokenizer)
    
    dump(corpus, open('corpus/MicroCorpus_512.pkl', 'wb'))
    
    # human corpus
    # meta = pd.read_csv('~/data5/download/MGnify/metadata.csv', index_col=0)
    # meta = meta['Env'].str.split(':', expand=True)[1]
    # meta = meta[meta == 'Host-associated']
    # human_abu = abu.loc[abu.index.isin(meta.index)]
    # human_corpus = MicroCorpus(abu=human_abu, tokenizer=tokenizer, preprocess=False)
    # dump(human_corpus, open('corpus/MicroCorpus_human_512.pkl', 'wb'))
    
    # microbes = abu.columns.tolist()
    # key_list.extend(microbes)
    
    # # build token dict
    # token_dict = {}
    # for i, key in enumerate(key_list):
    #     token_dict[key] = i
    # dump(token_dict, open('token_dict.pkl', 'wb'))
    
    # # calculate none zero median value of each microbe
    # median_dict = {}
    # for microbe in microbes:
    #     median_dict[microbe] = abu[microbe].replace(0, np.nan).median()
    # dump(median_dict, open('median_dict.pkl', 'wb'))