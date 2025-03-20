import pandas as pd
import numpy as np
import os
import torch
from torch.utils.data import Dataset

# class MicroTimeSeriesDataset(Dataset):
#     def __init__(self,
#                  abu: pd.DataFrame, # abundance table, rows are samples, columns are features
#                  subject_id: pd.Series, # The subject id to which each sample belongs
#                  timepoints: pd.Series, # timepoints for each sample
#                  mlm: bool = False, # whether to train a masked language model、
#                  mlm_rate: float = 0.15, # the rate of masking
#                  return_labels: bool = False # whether to return labels for masked language model
#                  ):
#         self.meta = pd.DataFrame({'subject_id': subject_id, 'timepoint': timepoints})
#         self.abu = abu.loc[self.meta.index]
#         self.num_samples = len(self.meta)
#         self.num_subjects = len(self.meta['subject_id'].unique())
#         self.max_timepoints = self.meta['timepoint'].max()
#         self.min_timepoints = self.meta['timepoint'].min()
        
#         if self.min_timepoints == 0:
#             print('Timepoints start from 0 which is used for padding, add 1 to all timepoints')
#             self.meta['timepoint'] += 1
#             self.max_timepoints += 1
        
#         self.mlm = mlm
#         self.mlm_rate = mlm_rate
#         self.return_labels = return_labels
        
#         self.subject_abus = {}
#         self.subject_timepoints = {}
        
#         for subject_id in self.meta['subject_id'].unique():
#             subject_abu, subject_timepoints = self._process_subject(subject_id)
#             self.subject_abus[subject_id] = subject_abu
#             self.subject_timepoints[subject_id] = subject_timepoints
            
#         self.input_values = torch.stack([self.subject_abus[subject_id] for subject_id in self.subject_abus])
#         self.position_ids = torch.stack([self.subject_timepoints[subject_id] for subject_id in self.subject_timepoints])
        
        
#     def __getitem__(self, idx):
#         input_values = self.input_values[idx]
#         position_ids = self.position_ids[idx]
#         attention_mask = torch.ones_like(position_ids)
#         attention_mask[position_ids == 0] = 0
        
#         labels = input_values.clone()
        
#         if self.mlm:   # masked language model
#             # masking
#             mask = torch.full(labels.shape[:-1], self.mlm_rate)
#             mask = mask * attention_mask    # only mask non-padding tokens
            
#             mask = torch.bernoulli(mask).bool()
#             labels[~mask] = -100
            
#             # 80% of the time, we replace masked input values with all-zero
#             indices_replaced = torch.bernoulli(torch.full(labels.shape[:-1], 1.)).bool() & mask
#             input_values[indices_replaced] = torch.zeros_like(input_values[indices_replaced])
            

#             # 10% of the time, we replace masked input tokens with random 
#             # indices_random = torch.bernoulli(torch.full(labels.shape[:-1], 0.5)).bool() & mask & ~indices_replaced
#             # random_words = torch.rand(labels.shape, dtype=torch.double)
#             # input_values[indices_random] = random_words[indices_random]
                
#             # The rest of the time (10% of the time) we keep the masked input tokens unchanged
            
#             return {'input_values': input_values.type(torch.float),
#                     'position_ids': position_ids.type(torch.int),
#                     'attention_mask': attention_mask,
#                     'labels': labels.type(torch.float)}
            
#         if not self.return_labels:
            
#             return{
#                 'input_values': input_values.type(torch.float),
#                 'position_ids': position_ids.type(torch.int),
#                 'attention_mask': attention_mask
#             }

#         else:
#             return{
#                 'input_values': input_values.type(torch.float),
#                 'position_ids': position_ids.type(torch.int),
#                 'attention_mask': attention_mask,
#                 'labels': labels.type(torch.float)
#             }
        
#     def __len__(self):
#         return self.num_subjects
    
#     def _process_subject(self, subject_id):
#         subject_abu = self.abu[self.meta['subject_id'] == subject_id]
#         subject_abu = torch.tensor(subject_abu.values)
#         subject_timepoints = self.meta[self.meta['subject_id'] == subject_id]['timepoint']
#         subject_timepoints = torch.tensor(subject_timepoints.values)
#         return self._pad(subject_abu, subject_timepoints)
    
#     def _pad(self, subject_abu, subject_timepoints):
#         abu_pads = torch.zeros((self.max_timepoints-subject_timepoints.shape[0], subject_abu.shape[1]))
#         timepoints_pads = torch.zeros((self.max_timepoints-subject_timepoints.shape[0]))
#         return torch.cat([subject_abu, abu_pads], 0), torch.cat([subject_timepoints, timepoints_pads], 0)
    
class MicroTSDataset(Dataset):
    def __init__(self,
                abu: pd.DataFrame, # abundance table, rows are samples, columns are features
                subject_id: pd.Series, # The subject id to which each sample belongs
                timepoints: pd.Series, # timepoints for each sample
                forecast: bool = False, # whether to forecast future values
                future_steps: int = 1, # number of future steps to forecast
    ):
        super().__init__()
        self.abu = abu
        self.subject_id = subject_id
        self.timepoints = timepoints
        self.data = pd.concat([subject_id, timepoints, abu], axis=1)
        self.num_samples = len(subject_id.unique())
        
        # timeline & samples
        self.timelines = self._get_timeline()
        self.timeline = np.sort(self.timepoints.unique())
        self.samples = self.timelines.index
        self.features = self.abu.columns
        
        # align data
        self.aligned_data = self._align()
        self.observe = (~torch.isnan(self.aligned_data)).float()   # 1 for observed, 0 for missing
        self.aligned_data[torch.isnan(self.aligned_data)] = 0
        
        # forecast
        self.forecast = forecast
        self.future_steps = future_steps
        
        
    def _get_timeline(self):
        timeline = self.data.groupby(self.subject_id.name).apply(
            lambda x: x[self.timepoints.name].values
        )
        return timeline
    
    def _align(self):
        # align timelines against the timeline, if a timepoint is missing, fill with np.nan
        aligned_data = []
        for i, timeline in enumerate(self.timelines):
            aligned = np.full((len(self.timeline), self.abu.shape[1]), np.nan)    # (num_samples, num_timepoints, num_features)
            run_data = self.data[self.data[self.subject_id.name] == self.samples[i]]
            for j, timepoint in enumerate(timeline):
                idx = np.where(self.timeline == timepoint)[0][0]
                #print(run_data[run_data[self.timepoints.name] == timepoint].iloc[:, 2:].values.shape)
                aligned[idx] = run_data[run_data[self.timepoints.name] == timepoint].iloc[:, 2:].values
            aligned_data.append(aligned)
        
        return torch.tensor(aligned_data, dtype=torch.float)
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        past_values = self.aligned_data[idx]
        past_observed_mask = self.observe[idx]
        
        # forecast
        if self.forecast:
            if len(past_values.shape) == 2: # if only one sample
                future_values = past_values.clone()[-self.future_steps:]
                past_values = past_values[:-self.future_steps]
                past_observed_mask = past_observed_mask[:-self.future_steps]
            else:
                future_values = past_values[:, -self.future_steps:] # handle batch
                past_values = past_values[:, :-self.future_steps]
                past_observed_mask = past_observed_mask[:, :-self.future_steps]
            return {
                'past_values': past_values,
                'past_observed_mask': past_observed_mask,
                'future_values': future_values,
            }
        
        return {
            'past_values': self.aligned_data[idx],
            'past_observed_mask': self.observe[idx],
        }
        
        
                 
                 
                 
                 
if __name__ == '__main__':
    abu = pd.read_csv('data/binned_data/corpse_original.csv')
    subject_ids = abu['ID']
    timepoints = abu['days']
    abu = abu[abu.columns.difference(['ID', 'days'])]
    
    # dataset = MicroTimeSeriesDataset(abu, subject_ids, timepoints)
    dataset = MicroTSDataset(abu, subject_ids, timepoints)
    