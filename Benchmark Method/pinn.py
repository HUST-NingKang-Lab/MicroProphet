import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
import random
import matplotlib.pyplot as plt
from scipy.spatial import distance

os.chdir("/invasion")


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

set_seed(42)


os.makedirs("result/abundance", exist_ok=True)
os.makedirs("result/glv", exist_ok=True)
os.makedirs("result/model", exist_ok=True)


def load_train_test_data(train_path, test_path, known_steps=4):
    def process_file(file_path):
        data = pd.read_csv(file_path)
        microbe_columns = data.columns[2:]  
        input_size = len(microbe_columns)
        subjects = data['subject_id'].unique()

        input_data = []
        target_data = []
        subject_ids = []
        time_stamps = []

        for subject in subjects:
            patient_df = data[data['subject_id'] == subject].sort_values(by='time')
            patient_values = patient_df.iloc[:, 2:].values  
            patient_time = patient_df['time'].values  

            if patient_values.shape[0] > known_steps:
                input_data.append(patient_values[:known_steps, :])
                target_data.append(patient_values[known_steps:, :])
                subject_ids.append([subject] * (patient_values.shape[0] - known_steps))
                time_stamps.append(patient_time[known_steps:])

        return input_data, target_data, subject_ids, time_stamps, input_size, microbe_columns

    train_result = process_file(train_path)
    test_result = process_file(test_path)

   
    assert train_result[4] == test_result[4]
    assert train_result[5].equals(test_result[5])
    return train_result[:4], test_result[:4], train_result[4], train_result[5]


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x, steps_to_predict):
        outputs = []
        h, c = None, None 
        
        for _ in range(steps_to_predict):
            out, (h, c) = self.lstm(x if len(outputs) == 0 else outputs[-1].unsqueeze(1), (h, c) if h is not None else None)
            out = self.fc(out[:, -1, :]) 
            out = self.relu(out)
            outputs.append(out)
        
        return torch.stack(outputs, dim=1)  

def calculate_glv_residual(abundances, interactions, ri, time_increment):
    if isinstance(interactions, np.ndarray):
        interactions = torch.tensor(interactions, dtype=torch.float32, device=abundances.device)
    if isinstance(ri, np.ndarray):
        ri = torch.tensor(ri, dtype=torch.float32, device=abundances.device)
    
    batch_size, time_steps, num_species = abundances.shape
    residuals = torch.zeros_like(abundances)
    
    for t in range(time_steps):
        for i in range(num_species):
            interaction_row = interactions[i].unsqueeze(0)
            change_per_capita = (ri[i] + torch.sum(abundances[:, t, :] * interaction_row, dim=1)) * time_increment
            change_per_capita = change_per_capita.view(-1)
            predicted_change = abundances[:, t, i] * change_per_capita
            residuals[:, t, i] = predicted_change

    return residuals


def composite_loss(outputs, targets, abundances, interactions, ri, time_increment, data_weight=1.0, physics_weight=1.0):
    data_loss = nn.MSELoss()(outputs, targets)
    glv_residuals = calculate_glv_residual(abundances, interactions, ri, time_increment)
    physics_loss = torch.mean(glv_residuals ** 2)
    total_loss = data_weight * data_loss + physics_weight * physics_loss
    return total_loss


class MicrobiomeDataset(Dataset):
    def __init__(self, inputs, targets):
        self.inputs = [torch.tensor(item, dtype=torch.float32) for item in inputs]
        self.targets = [torch.tensor(item, dtype=torch.float32) for item in targets]
    
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]


def train_model(model, train_loader, optimizer, interactions, ri, time_increment, microbe_columns, epochs=50, data_weight=1.0, physics_weight=1.0):
    model.train()
    losses = []
    os.makedirs("result/abundance/train", exist_ok=True)  
    os.makedirs("result/abundance/test", exist_ok=True)
    for epoch in range(epochs):
        total_loss = 0
        predictions = []
        actuals = []
        
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs, steps_to_predict=targets.size(1))
            loss = composite_loss(outputs, targets, inputs, interactions, ri, time_increment, data_weight, physics_weight)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
            predictions.append(outputs.detach().numpy())
            actuals.append(targets.numpy())
        
        avg_loss = total_loss / len(train_loader)
        losses.append(avg_loss)
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}')
        
        
        for i, (pred, actual) in enumerate(zip(predictions, actuals)):
            pred_2d = pred.reshape(-1, pred.shape[-1])
            actual_2d = actual.reshape(-1, actual.shape[-1])
            
            pred_df = pd.DataFrame(pred_2d, columns=microbe_columns)
            actual_df = pd.DataFrame(actual_2d, columns=microbe_columns)
            
            pred_file = f'result/abundance/train/patient_{i+1}_epoch_{epoch+1}_predicted.csv'
            actual_file = f'result/abundance/train/patient_{i+1}_epoch_{epoch+1}_actual.csv'
            pred_df.to_csv(pred_file, index=False)
            actual_df.to_csv(actual_file, index=False)
    
    plt.figure()
    plt.plot(range(1, epochs + 1), losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Epochs')
    plt.legend()
    plt.savefig("result/loss.png")
    print("Loss curve saved to: result/loss.png")

def predict_and_save(model, test_loader, microbe_columns, test_subject_ids, test_times, save_path="result/model/trained_lstm_model_with_glv.pth"):
    model.load_state_dict(torch.load(save_path))
    model.eval()
    predictions = []
    actuals = []
    subject_ids = []
    times = []

    with torch.no_grad():
        for i, (inputs, targets) in enumerate(test_loader):
            outputs = model(inputs, steps_to_predict=targets.size(1))
            predictions.append(outputs.numpy())
            actuals.append(targets.numpy())
            subject_ids.extend(test_subject_ids[i])
            times.extend(test_times[i])

    pred_array = np.concatenate([pred.reshape(-1, pred.shape[-1]) for pred in predictions], axis=0)
    actual_array = np.concatenate([act.reshape(-1, act.shape[-1]) for act in actuals], axis=0)

    pred_df = pd.DataFrame(pred_array, columns=microbe_columns)
    actual_df = pd.DataFrame(actual_array, columns=microbe_columns)
    pred_df.insert(0, "time", times)
    pred_df.insert(0, "subject_id", subject_ids)
    actual_df.insert(0, "time", times)
    actual_df.insert(0, "subject_id", subject_ids)

    pred_df.to_csv("result/abundance/invasion_pred.csv", index=False)
    actual_df.to_csv("result/abundance/invasion_labels.csv", index=False)

    print("Prediction and true abundance saved to 'result/abundance/pred.csv' and 'result/abundance/true.csv'.")


def save_growth_and_interactions(ri, interactions, microbe_columns):
    ri_df = pd.DataFrame(ri, index=microbe_columns, columns=['Intrinsic Growth Rate'])
    ri_file = 'result/glv/intrinsic_growth_rates.csv'
    ri_df.to_csv(ri_file)
    print("Intrinsic Growth Rates saved to:", ri_file)
    
    interactions = (interactions + interactions.T) / 2
    np.fill_diagonal(interactions, 0)
 
    interactions_df = pd.DataFrame(interactions, index=microbe_columns, columns=microbe_columns)
    interactions_file = 'result/glv/species_interactions.csv'
    interactions_df.to_csv(interactions_file)
    print("Species Interactions Matrix saved to:", interactions_file)


def main():
    hidden_size = 32
    batch_size = 32
    learning_rate = 0.001
    epochs = 1000
    time_increment = 0.1  
    known_steps = 4  

    num_species = 20
    interaction_strength = 0.1
    growth_rate_range = (-1, 1)

    interactions = np.random.uniform(-interaction_strength, interaction_strength, (num_species, num_species))
    np.fill_diagonal(interactions, 0)
    ri = np.random.uniform(growth_rate_range[0], growth_rate_range[1], num_species)

    train_file = 'invasion_train.csv'
    test_file = 'invasion_test.csv'

    train_data, test_data, input_size, microbe_columns = load_train_test_data(
        train_file, test_file, known_steps=known_steps)

    X_train, Y_train, train_subject_ids, train_times = train_data
    X_test, Y_test, test_subject_ids, test_times = test_data

    train_dataset = MicrobiomeDataset(X_train, Y_train)
    test_dataset = MicrobiomeDataset(X_test, Y_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    model = LSTMModel(input_size=input_size, hidden_size=hidden_size, output_size=input_size)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model_save_path = "result/model/trained_lstm_model_with_glv.pth"
    train_model(model, train_loader, optimizer, interactions, ri, time_increment, microbe_columns, epochs=epochs)

    torch.save(model.state_dict(), model_save_path)
    predict_and_save(model, test_loader, microbe_columns, test_subject_ids, test_times, save_path=model_save_path)

    save_growth_and_interactions(ri, interactions, microbe_columns)

if __name__ == "__main__":
    main()
