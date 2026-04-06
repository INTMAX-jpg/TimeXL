import os
import torch
import torch.nn.functional as F
import logging
import sys
import numpy as np
from torch.utils.data import DataLoader

# Add repo root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_ROOT)

from data.real_data_loader import ProcessedWeatherDataset
from training.models import TimeXLModel, PrototypeManager
from training.loss import TimeXLLoss
from training.test_and_eval.eval_metrics_cal import calculate_metrics

# Configure Logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("Evaluator")

def evaluate_encoder():
    logger.info("=== Starting Detailed Evaluation ===")
    
    # Configuration
    DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
    CITY = 'San Francisco'
    BATCH_SIZE = 64
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    logger.info(f"Device: {DEVICE}")
    
    # 1. Load Data
    try:
        test_dataset = ProcessedWeatherDataset(DATA_DIR, city=CITY, split='test')
    except FileNotFoundError as e:
        logger.error(f"Data not found: {e}")
        return

    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    num_classes = len(test_dataset.label_map)
    
    # 2. Load Model
    k = 10 
    model = TimeXLModel(num_classes=num_classes, k=k)
    pm = PrototypeManager(num_classes=num_classes, k=k)
    loss_fn = TimeXLLoss()
    
    best_encoder_path = os.path.join(PROJECT_ROOT, 'pth', 'best_encoder.pth')
    best_prototypes_path = os.path.join(PROJECT_ROOT, 'pth', 'best_prototypes.pth')
    if os.path.exists(best_encoder_path):
        logger.info("Loading best model checkpoint...")
        model.load_state_dict(torch.load(best_encoder_path, map_location=DEVICE))
        pm.load_state_dict(torch.load(best_prototypes_path, map_location=DEVICE))
    else:
        logger.error("Checkpoint not found! Please run train_encoder.py first.")
        return

    model.to(DEVICE)
    pm.to(DEVICE)
    model.eval()
    
    # 3. Evaluation Loop
    total_loss = 0.0
    total_kl_loss = 0.0
    total_clst_loss = 0.0
    total_evi_loss = 0.0
    total_div_loss = 0.0
    
    total_samples = 0
    correct_predictions = 0
    
    # For metric calculation (avg across all samples)
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in test_loader:
            x_time, original_texts, targets = batch
            x_time = x_time.to(DEVICE)
            targets = targets.to(DEVICE)
            
            # Forward
            z_time = model.time_encoder(x_time)
            z_text = model.text_encoder(original_texts)
            sim_time, sim_text = pm(z_time, z_text)
            predictions = model.fusion(sim_time, sim_text)
            
            # Loss Calculation
            loss, loss_details = loss_fn(
                predictions, targets, z_time, z_text, pm
            )
            
            # Aggregate Losses
            batch_size = targets.size(0)
            total_loss += loss.item() * batch_size
            total_kl_loss += loss_details['kl'] * batch_size
            total_clst_loss += loss_details['clst'] * batch_size
            total_evi_loss += loss_details['evi'] * batch_size
            total_div_loss += loss_details['div'] * batch_size
            
            # Accuracy
            _, predicted_indices = torch.max(predictions, 1)
            if targets.dim() > 1 and targets.size(1) > 1:
                target_indices = torch.argmax(targets, dim=1)
            else:
                target_indices = targets
            
            correct_predictions += (predicted_indices == target_indices).sum().item()
            total_samples += batch_size
            
            # Collect for Metrics
            probs = F.softmax(predictions, dim=1)
            all_preds.append(probs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    # 4. Final Calculations
    avg_loss = total_loss / total_samples
    avg_kl = total_kl_loss / total_samples
    avg_clst = total_clst_loss / total_samples
    avg_evi = total_evi_loss / total_samples
    avg_div = total_div_loss / total_samples
    accuracy = correct_predictions / total_samples
    
    # Calculate Distribution Metrics (Avg KL and MSE per sample)
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    dist_metrics_kl = []
    dist_metrics_mse = []
    dist_metrics_mae = []
    
    for p, t in zip(all_preds, all_targets):
        m = calculate_metrics(p, t)
        dist_metrics_kl.append(m['kl'])
        dist_metrics_mse.append(m['mse'])
        dist_metrics_mae.append(np.mean(np.abs(p - t)))
        
    avg_dist_kl = np.mean(dist_metrics_kl)
    avg_dist_mse = np.mean(dist_metrics_mse)
    avg_dist_mae = np.mean(dist_metrics_mae)
    
    # 5. Output Report
    print("\n" + "="*50)
    print("           TEST SET EVALUATION REPORT           ")
    print("="*50)
    print(f"Total Samples: {total_samples}")
    print(f"Accuracy:      {accuracy:.4f} ({accuracy*100:.2f}%)")
    print("-" * 50)
    print("LOSS BREAKDOWN (Average per sample):")
    print(f"Total Loss:         {avg_loss:.6f}")
    print(f"KL Loss (Training): {avg_kl:.6f}  <-- Used in optimization")
    print(f"Clustering Loss:    {avg_clst:.6f}")
    print(f"Evidence Loss:      {avg_evi:.6f}")
    print(f"Diversity Loss:     {avg_div:.6f}")
    print("-" * 50)
    print("DISTRIBUTION METRICS (Direct Calculation):")
    print(f"Avg KL Divergence:  {avg_dist_kl:.6f}")
    print(f"Avg MAE:            {avg_dist_mae:.6f}")
    print(f"Avg MSE:            {avg_dist_mse:.6f}")
    print("="*50 + "\n")

if __name__ == "__main__":
    evaluate_encoder()
