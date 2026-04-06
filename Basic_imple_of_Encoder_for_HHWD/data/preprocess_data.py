import os
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm

# Configuration
DATA_DIR = '/root/timexl_repo/data/historical-hourly-weather-data'
SAVE_DIR = '/root/timexl_repo/data/processed_data'
CITY = 'San Francisco'
WINDOW_SIZE = 24
PRED_HORIZON = 24

def get_simplified_label(desc: str) -> str:
    """Map detailed description to simplified category"""
    desc = str(desc).lower()
    
    # Snow / Ice categories
    if any(x in desc for x in ['snow', 'sleet', 'ice', 'hail', 'freezing']):
        return 'Snow'
        
    # Rain / Precipitation categories
    if any(x in desc for x in ['rain', 'drizzle', 'shower', 'thunderstorm', 'storm', 'squall']):
        return 'Rain'
        
    # Everything else
    return 'No Precipitation'

def process_data():
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    print(f"Loading data for {CITY} from {DATA_DIR}...")
    
    # Load all required files
    files = {
        'humidity': 'humidity.csv',
        'pressure': 'pressure.csv',
        'temperature': 'temperature.csv',
        'wind_direction': 'wind_direction.csv',
        'wind_speed': 'wind_speed.csv',
        'description': 'weather_description.csv'
    }
    
    dfs = {}
    for key, filename in files.items():
        path = os.path.join(DATA_DIR, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"File {filename} not found in {DATA_DIR}")
        dfs[key] = pd.read_csv(path)

    # Extract city data and handle missing values
    data = {}
    for key, df in dfs.items():
        # Forward fill then backward fill
        series = df[CITY].ffill().bfill()
        data[key] = series.values

    # Normalize numerical features
    # Features: humidity, wind_speed, pressure, temperature, wind_direction
    feature_keys = ['humidity', 'wind_speed', 'pressure', 'temperature', 'wind_direction']
    numerical_data = []
    
    print("Normalizing numerical features...")
    for key in feature_keys:
        vals = data[key].astype(float)
        mean = np.mean(vals)
        std = np.std(vals) + 1e-8
        norm_vals = (vals - mean) / std
        numerical_data.append(norm_vals)
    
    # Stack features: [Time, Features]
    # Shape: [Total_Time, 5]
    numerical_features = np.stack(numerical_data, axis=1)
    
    # Text data
    text_data = data['description']
    
    # Prepare samples
    num_samples = len(text_data) - WINDOW_SIZE - PRED_HORIZON
    print(f"Total time steps: {len(text_data)}")
    print(f"Generating {num_samples} samples with Window={WINDOW_SIZE}, Horizon={PRED_HORIZON}...")
    
    samples = []
    
    # Categories map
    cat_map = {'No Precipitation': 0, 'Rain': 1, 'Snow': 2}
    
    for i in tqdm(range(num_samples)):
        # 1. Input Window [i : i+24]
        # Numerical X: [24, 5]
        num_x = numerical_features[i : i + WINDOW_SIZE]
        num_x_tensor = torch.tensor(num_x, dtype=torch.float32)
        
        # Text X: List[str] length 24
        text_x = text_data[i : i + WINDOW_SIZE].tolist()
        
        # 2. Target Window [i+24 : i+48]
        # Calculate distribution label
        target_descs = text_data[i + WINDOW_SIZE : i + WINDOW_SIZE + PRED_HORIZON]
        
        counts = {0: 0, 1: 0, 2: 0} # No Precip, Rain, Snow
        
        for desc in target_descs:
            simple_label = get_simplified_label(desc)
            cat_idx = cat_map[simple_label]
            counts[cat_idx] += 1
            
        # Normalize to probability distribution
        # Total count is PRED_HORIZON (24)
        probs = [counts[0]/PRED_HORIZON, counts[1]/PRED_HORIZON, counts[2]/PRED_HORIZON]
        label_y = torch.tensor(probs, dtype=torch.float32)
        
        # Verify sum is 1.0 (approx)
        if abs(sum(probs) - 1.0) > 1e-5:
            print(f"Warning: Probs sum to {sum(probs)} at index {i}")
            
        samples.append({
            "numerical_x": num_x_tensor,
            "text_x": text_x,
            "label_y": label_y
        })
        
    # Split Data (80/10/10)
    total = len(samples)
    train_end = int(total * 0.8)
    val_end = int(total * 0.9)
    
    train_samples = samples[:train_end]
    val_samples = samples[train_end:val_end]
    test_samples = samples[val_end:]
    
    print(f"Train samples: {len(train_samples)}")
    print(f"Val samples: {len(val_samples)}")
    print(f"Test samples: {len(test_samples)}")
    
    # Save
    print("Saving datasets...")
    torch.save(train_samples, os.path.join(SAVE_DIR, f'processed_train_{CITY.replace(" ", "_")}.pt'))
    torch.save(val_samples, os.path.join(SAVE_DIR, f'processed_val_{CITY.replace(" ", "_")}.pt'))
    torch.save(test_samples, os.path.join(SAVE_DIR, f'processed_test_{CITY.replace(" ", "_")}.pt'))
    
    print("Done!")

if __name__ == '__main__':
    process_data()
