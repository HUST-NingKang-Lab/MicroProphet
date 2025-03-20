import os
import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import joblib

base_dir = "/invasion"
os.chdir(base_dir)
data_name = os.path.basename(base_dir)  # Extract folder name as data name

def set_seed(seed=0):
    np.random.seed(seed)

set_seed(0)

os.makedirs("result/abundance", exist_ok=True)
os.makedirs("result/model", exist_ok=True)

def load_train_test_data(train_path, test_path, known_steps=3):
    def process_file(file_path):
        data = pd.read_csv(file_path)
        microbe_columns = data.columns[2:]  # Microbe data columns
        input_size = len(microbe_columns)
        subjects = data['subject_id'].unique()

        input_data = []
        target_data = []
        subject_ids = []
        time_stamps = []

        for subject in subjects:
            patient_df = data[data['subject_id'] == subject].sort_values(by='time')
            patient_values = patient_df.iloc[:, 2:].values  # Microbe columns
            patient_time = patient_df['time'].values  # Time column

            if patient_values.shape[0] > known_steps:
                input_data.append(patient_values[:known_steps, :].flatten())  # Flatten input
                target_data.append(patient_values[known_steps:, :].flatten())  # Flatten target
                subject_ids.append([subject] * (patient_values.shape[0] - known_steps))
                time_stamps.append(patient_time[known_steps:])

        return input_data, target_data, subject_ids, time_stamps, input_size, microbe_columns

    train_result = process_file(train_path)
    test_result = process_file(test_path)

    # Ensure the microbe columns and size are consistent between train and test
    assert train_result[4] == test_result[4]
    assert train_result[5].equals(test_result[5])

    print("Loading completed:")
    print(f"Training samples: {len(train_result[0])}, Testing samples: {len(test_result[0])}")

    return train_result[:4], test_result[:4], train_result[4], train_result[5]

def train_lasso(X_train, Y_train, model_save_path, alpha=1.0):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    
    model = Lasso(alpha=alpha, random_state=0)
    model.fit(X_train, Y_train)

    # Save the model and scaler
    joblib.dump(model, model_save_path)
    joblib.dump(scaler, model_save_path.replace('.joblib', '_scaler.joblib'))

    print(f"Model saved to: {model_save_path}")
    return model, scaler

def evaluate_and_save_predictions(model, scaler, X_test, Y_test, test_subject_ids, test_times, microbe_columns):
    X_test = scaler.transform(X_test)
    predictions = model.predict(X_test)

    # Reshape to match the microbe columns
    predictions = predictions.reshape(-1, len(microbe_columns))
    Y_test = Y_test.reshape(-1, len(microbe_columns))

    # Ensure test_times and test_subject_ids are 1D arrays
    test_times = np.concatenate(test_times) if isinstance(test_times, list) else test_times
    test_subject_ids = np.concatenate(test_subject_ids) if isinstance(test_subject_ids, list) else test_subject_ids

    # Create DataFrame
    pred_df = pd.DataFrame(predictions, columns=microbe_columns)
    actual_df = pd.DataFrame(Y_test, columns=microbe_columns)
    pred_df.insert(0, "time", test_times)
    pred_df.insert(0, "subject_id", test_subject_ids)
    actual_df.insert(0, "time", test_times)
    actual_df.insert(0, "subject_id", test_subject_ids)

    # Save files
    pred_df.to_csv(f"result/abundance/{data_name}_pred.csv", index=False)
    actual_df.to_csv(f"result/abundance/{data_name}_labels.csv", index=False)

    print(f"Predictions saved to: result/abundance/{data_name}_pred.csv")
    print(f"True labels saved to: result/abundance/{data_name}_true.csv")

    mse = mean_squared_error(Y_test, predictions)
    print(f"Test MSE: {mse:.4f}")

def main():
    known_steps = 4  # Number of time steps used for training

    train_file = f'{data_name}_train.csv'
    test_file = f'{data_name}_test.csv'

    train_data, test_data, input_size, microbe_columns = load_train_test_data(
        train_file, test_file, known_steps=known_steps)

    X_train, Y_train, train_subject_ids, train_times = train_data
    X_test, Y_test, test_subject_ids, test_times = test_data

    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    X_test = np.array(X_test)
    Y_test = np.array(Y_test)

    model_save_path = f"result/model/trained_{data_name}_lasso.joblib"

    model, scaler = train_lasso(X_train, Y_train, model_save_path, alpha=0.1)  # You can adjust alpha

    evaluate_and_save_predictions(model, scaler, X_test, Y_test, test_subject_ids, test_times, microbe_columns)

if __name__ == "__main__":
    main()
