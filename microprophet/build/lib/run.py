import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
import shap
import torch.nn.functional as F
from torch.utils.data import random_split, DataLoader
from include.TimeSeries import MicroTSDataset
from include.utils import seed_everything
import matplotlib.pyplot as plt
from transformers import (
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    PatchTSTConfig,
    PatchTSTForPrediction,
)
import warnings
warnings.filterwarnings("ignore")

seed_everything(22)
def train_patchtst_on_dataset(train_set, 
                              val_set, 
                              test_set, 
                              num_input_channels,
                              context_length,
                              future_steps, 
                              model_save_path,
                              args):
    config = {
        'num_input_channels': num_input_channels,
        'context_length': context_length,
        'do_mask_input': True,
        'num_hidden_layers': 3,
        'd_model': 64,
        'patch_length': 1,
        'patch_stride': 1,
        'nhead': 4,
        'ffn_dim': 64*4,
        'scaling': False,
        'prediction_length': future_steps,
    }
    model = PatchTSTForPrediction(PatchTSTConfig(**config))
    training_args = {
        'do_train': True,
        'do_eval': True,
        'disable_tqdm': False,
        'lr_scheduler_type': 'linear',
        'per_device_train_batch_size': 8,
        'num_train_epochs': 1000,
        'evaluation_strategy': 'epoch',
        'save_strategy': 'epoch',
        'label_names': ['past_values'],
        'logging_steps': 1,
        'output_dir': model_save_path,
        'logging_dir': 'transformers_logs',
        'load_best_model_at_end': True,
    }
    training_args = TrainingArguments(**training_args)

    callback = EarlyStoppingCallback(early_stopping_patience=3)

    from torch.utils.data import Subset
    indices = [0, 2]  
    subset = Subset(train_set, indices)
    for i in range(len(subset)):
        print("past_values", subset[i]['past_values'].shape)
        print("past_observed_mask", subset[i]['past_observed_mask'])
        print("future_values", subset[i]['future_values'].shape)


    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_set,
        eval_dataset=val_set,
        callbacks=[callback],
    )
    trainer.train()
    if model_save_path and args.export_model:
        trainer.save_model(args.export_path+"model")

    predictions = trainer.predict(test_set)
    pred = predictions.predictions[0]
    pred = F.relu(torch.tensor(pred)).numpy()  
    labels = torch.stack([batch['future_values'] for batch in test_set]).numpy()
    return model, test_set, pred, labels

class WrapperModel(nn.Module):
    def __init__(self, model, target_timestep=0):
        super(WrapperModel, self).__init__()
        self.model = model
        self.target_timestep = target_timestep

    def forward(self, x):
        past_values, past_observed_mask = torch.chunk(x, chunks=2, dim=1)  
        output = self.model(past_values, past_observed_mask).prediction_outputs  
        return torch.mean(output, dim=1).squeeze()


def analyze_feature_importance(model, test_set, dataset, device='cpu', target_timestep=5):
    """使用 SHAP 计算 PatchTST 模型的特征重要性，并输出表格"""
    past_values = []
    past_observed_mask = []

    for i in range(len(test_set)):
        sample = test_set[i]
        past_values.append(sample['past_values'].unsqueeze(0))
        past_observed_mask.append(sample['past_observed_mask'].unsqueeze(0))

    past_values = torch.cat(past_values, dim=0).to(device)
    past_observed_mask = torch.cat(past_observed_mask, dim=0).to(device)

    my_model = WrapperModel(model, target_timestep=target_timestep).to(device)
    my_model.eval()

    baseline_data = torch.cat([past_values, past_observed_mask], dim=1)
    explainer = shap.GradientExplainer(my_model, baseline_data)

    shap_values = explainer.shap_values(baseline_data)
    shap_tensors = torch.stack([torch.tensor(arr) for arr in shap_values])
    shap_mean = torch.mean(shap_tensors, dim=0)

    feature_importance = torch.mean(shap_mean, dim=(0, 1))

    feature_names = dataset.features
    shap_df = pd.DataFrame({
        'Feature': feature_names,
        'SHAP Importance': feature_importance.cpu().numpy()
    }).sort_values(by='SHAP Importance', ascending=False)

    return shap_df

def visual_attent_score(model, test_set,args, device='cpu', target_timestep=5):
    past_values = []
    past_observed_mask = []

    for i in range(len(test_set)):
        sample = test_set[i]
        past_values.append(sample['past_values'].unsqueeze(0)) 
        past_observed_mask.append(sample['past_observed_mask'].unsqueeze(0))

    past_values = torch.cat(past_values, dim=0).to(device)
    past_observed_mask = torch.cat(past_observed_mask, dim=0).to(device)
 
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        attention_score = model(past_values, past_observed_mask, output_attentions=True).attentions
    attention_score = attention_score[2]
    attention_scores_np = attention_score[0][0].numpy()

    plt.figure(figsize=(8, 8))
    plt.imshow(attention_scores_np, cmap='viridis', vmin=0, vmax=1)  
    plt.colorbar(label='Attention Score')  
    plt.title("Attention Scores")
    
    time_steps = range(len(test_set[0]['past_values'])) 
    plt.xlabel("Time Step")
    plt.ylabel("Time Step")


    for i in range(attention_scores_np.shape[0]):
        for j in range(attention_scores_np.shape[0]):
            plt.text(j, i, f"{attention_scores_np[i, j]:.2f}", ha="center", va="center", color="w")

    plt.xticks(ticks=np.arange(len(time_steps)), labels=[f"t{i}" for i in time_steps])
    plt.yticks(ticks=np.arange(len(time_steps)), labels=[f"t{i}" for i in time_steps])

    plt.savefig(f'{args.export_path}heatmap_0.png', format='png')

def calculate_metrics(pred, labels):
    mse = np.mean((pred - labels) ** 2)
    corr = np.corrcoef(pred.flatten(), labels.flatten())[0, 1]
    return mse, corr

def load_data(file_path):
    data = pd.read_csv(file_path)
    subject_ids = data['subject_id']
    timepoints = data['time']
    abu = data[data.columns.difference(['subject_id', 'time'])]
    return data, abu, subject_ids, timepoints

def create_dataset(data, subject_ids, timepoints):
    dataset = MicroTSDataset(data, subject_ids, timepoints)
    dataset.forecast = True
    dataset.future_steps = int(0.7 * dataset.timeline.shape[0])
    return dataset

def create_results(dataset, pred, labels, test_idx):
    pred = pred.reshape(-1, dataset.features.shape[0])
    labels = labels.reshape(-1, dataset.features.shape[0])
    pred_df = pd.DataFrame({
        'subject_id': [id for id in dataset.samples[test_idx] for _ in range(dataset.timeline[-dataset.future_steps:].shape[0])],
        'time': [timepoint for _ in dataset.samples[test_idx] for timepoint in dataset.timeline[-dataset.future_steps:]],
    })
    pred_df = pd.concat([pred_df, pd.DataFrame(pred, columns=dataset.features)], axis=1)
    labels_df = pd.DataFrame({
        'subject_id': [id for id in dataset.samples[test_idx] for _ in range(dataset.timeline[-dataset.future_steps:].shape[0])],
        'time': [timepoint for _ in dataset.samples[test_idx] for timepoint in dataset.timeline[-dataset.future_steps:]],
    })
    labels_df = pd.concat([labels_df, pd.DataFrame(labels, columns=dataset.features)], axis=1)
    
    mae, corr = calculate_metrics(pred, labels)
    return pred_df, labels_df, mae, corr

def run(args):
    dataset_name=os.path.basename(args.data_path).split(".")[0]
    data, abu, subject_ids, timepoints = load_data(args.data_path)
    dataset = create_dataset(abu, subject_ids, timepoints)

    dt_size = len(dataset)
    train_size = int(dt_size * 0.7)
    val_size = dt_size - train_size

    train_set_original, test_set = random_split(dataset, [train_size, val_size])
    train_samples = dataset.samples[train_set_original.indices]
    test_samples = dataset.samples[test_set.indices]
    if args.export_split or args.mode=="split":
        data[data['subject_id'].isin(train_samples)].to_csv(args.export_path+dataset_name+"_train.csv", index=False)
        data[data['subject_id'].isin(test_samples)].to_csv(args.export_path+dataset_name+"_test.csv", index=False)
        if args.mode=="split":
            print(f"分割完成。导出到{args.export_path}")
            exit()


    dt_size = len(train_set_original)
    train_size = int(dt_size * 0.7)
    val_size = dt_size - train_size
    train_set, val_set = random_split(train_set_original, [train_size, val_size])
    if args.export_model and not os.path.exists(f"{args.export_path}model"):
        os.makedirs(f"{args.export_path}model")

    model, test_set, pred, labels = train_patchtst_on_dataset(train_set,
                                                val_set,
                                                test_set,
                                                dataset.features.shape[0],
                                                dataset.timeline.shape[0] - dataset.future_steps,
                                                dataset.future_steps,
                                                f"{args.export_path}train_checkpoints",
                                                args)

    pred_df, labels_df, mse, corr = create_results(dataset, pred, labels, test_set.indices)
    
    model, test_set, pred_train, labels_train = train_patchtst_on_dataset(train_set,
                                                val_set,
                                                train_set_original,
                                                dataset.features.shape[0],
                                                dataset.timeline.shape[0] - dataset.future_steps,
                                                dataset.future_steps,
                                                f"{args.export_path}train_checkpoints",
                                                args)


    if args.mode=="shap":
        shap_df=analyze_feature_importance(model, test_set, dataset)
        shap_df.to_csv(f"{args.export_path}shap_feature_importance.csv",index=False)
        exit()

    if args.mode=="attention":
        visual_attent_score(model, test_set,args)
        exit()
   
    pred_df_train, labels_df_train, mse_train, corr_train = create_results(dataset, pred_train, labels_train, train_set_original.indices)
  
    past = dataset.timeline.shape[0] - dataset.future_steps
    future = dataset.future_steps

    result_df = pd.DataFrame(columns=['mse', 'corr', 'past', 'future'])
    result_df = result_df._append({
                                       'mse': mse,
                                       'corr':corr,
                                       'past': past,
                                       'future':future}, ignore_index=True)
    pred_df.to_csv(f'{args.export_path+dataset_name}_pred.csv', index=False)
    labels_df.to_csv(f'{args.export_path+dataset_name}_labels.csv', index=False)
    pred_df_train.to_csv(f'{args.export_path+dataset_name}_train_pred.csv', index=False)
    labels_df_train.to_csv(f'{args.export_path+dataset_name}_train_labels.csv', index=False)
    result_df.to_csv(f'{args.export_path}results.csv', index=False)
