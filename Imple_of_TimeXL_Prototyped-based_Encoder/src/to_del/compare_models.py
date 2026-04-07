import torch
import torch.nn.functional as F
import os
import sys
import logging
import re
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_ROOT)

from training.models import TimeXLModel, PrototypeManager
from .real_data_loader import ProcessedWeatherDataset
from training.llm_agents import PredictionLLM
from training.base_trainer import BaseTrainer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def calculate_kl_divergence(pred_probs, true_probs):
    """
    Calculate KL Divergence between prediction and truth.
    KL(P || Q) = sum(P(x) * log(P(x) / Q(x)))
    Here we want to measure how well Prediction (Q) approximates Truth (P).
    Usually KL(Truth || Prediction).
    """
    # Avoid log(0)
    epsilon = 1e-8
    pred_probs = torch.clamp(pred_probs, min=epsilon)
    true_probs = torch.clamp(true_probs, min=epsilon)
    
    # KL(P || Q) = sum(P * log(P/Q)) = sum(P * (log P - log Q))
    kl = torch.sum(true_probs * (torch.log(true_probs) - torch.log(pred_probs)))
    return kl.item()

def parse_llm_prediction(llm_output):
    """
    Parse the LLM output string to extract probabilities.
    Expected format includes: "No Precipitation: 0.1, Rain: 0.9, Snow: 0.0"
    or similar key-value pairs.
    """
    try:
        # Simple parsing logic - can be improved based on actual LLM output format
        # Normalize text
        text = llm_output.lower().replace(',', '\n').replace(':', ' ')
        
        probs = {'no precipitation': 0.0, 'rain': 0.0, 'snow': 0.0}
        
        # Look for patterns like "category 0.5" or "category: 0.5"
        for line in text.split('\n'):
            line = line.strip()
            for key in probs.keys():
                if key in line:
                    # Extract number
                    match = re.search(r"(\d+(\.\d+)?)", line.replace(key, ''))
                    if match:
                        probs[key] = float(match.group(1))
        
        # Normalize to sum to 1.0
        total = sum(probs.values())
        if total > 0:
            return torch.tensor([probs['no precipitation'], probs['rain'], probs['snow']]) / total
        else:
            logger.warning(f"Could not parse probabilities from: {llm_output}")
            return torch.tensor([0.33, 0.33, 0.33]) # Fallback
            
    except Exception as e:
        logger.error(f"Error parsing LLM output: {e}")
        return torch.tensor([0.33, 0.33, 0.33])

def main():
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        print("\n" + "!"*80)
        print("ERROR: DEEPSEEK_API_KEY environment variable is not set.")
        print("Please set the API key to run real LLM predictions.")
        print("Example: export DEEPSEEK_API_KEY='your_key_here'")
        print("!"*80 + "\n")
        return

    DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
    CITY = 'San Francisco'
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load Data
    print("Loading test data...")
    test_dataset = ProcessedWeatherDataset(DATA_DIR, city=CITY, split='test')
    
    # Load Model
    print("Loading Encoder model...")
    num_classes = 3
    k = 10
    model = TimeXLModel(num_classes=num_classes, k=k).to(DEVICE)
    pm = PrototypeManager(num_classes=num_classes, k=k).to(DEVICE)
    
    try:
        best_encoder_path = os.path.join(PROJECT_ROOT, 'best_encoder.pth')
        best_prototypes_path = os.path.join(PROJECT_ROOT, 'best_prototypes.pth')
        model.load_state_dict(torch.load(best_encoder_path, map_location=DEVICE))
        pm.load_state_dict(torch.load(best_prototypes_path, map_location=DEVICE))
    except FileNotFoundError:
        print("Model checkpoints not found. Please run training first.")
        return

    # Initialize LLM
    print("Initializing Prediction LLM...")
    llm_agent = PredictionLLM(api_key=api_key)
    
    # Select 3 samples
    indices = [0, 10, 20] # Select 3 distinct samples
    
    print("\n" + "="*80)
    print(f"COMPARING ENCODER vs REAL LLM on {len(indices)} SAMPLES")
    print("="*80)
    
    total_encoder_kl = 0
    total_llm_kl = 0
    total_encoder_acc = 0
    total_llm_acc = 0
    
    # Prepare dummy similar cases for demonstration (using first 2 train samples)
    train_dataset_full = ProcessedWeatherDataset(DATA_DIR, city=CITY, split='train')
    case_explanations = []
    for train_idx in [0, 5]:
         _, t_text, t_target = train_dataset_full[train_idx]
         t_target_list = t_target.tolist()
         case_str = f"Input: {t_text[:50]}...\nTruth: No Precip: {t_target_list[0]:.2f}, Rain: {t_target_list[1]:.2f}, Snow: {t_target_list[2]:.2f}"
         case_explanations.append(case_str)
    
    for i, idx in enumerate(indices):
        if idx >= len(test_dataset):
            continue
            
        x_time, original_text, target = test_dataset[idx]
        target = target.to(DEVICE) # [3] probabilities
        
        # 1. Encoder Prediction
        model.eval()
        with torch.no_grad():
            x_time_batch = x_time.unsqueeze(0).to(DEVICE)
            z_time = model.time_encoder(x_time_batch)
            z_text = model.text_encoder([original_text])
            sim_time, sim_text = pm(z_time, z_text)
            encoder_logits = model.fusion(sim_time, sim_text)
            encoder_probs = F.softmax(encoder_logits, dim=1).squeeze(0)
            
        # 2. LLM Prediction
        print(f"\nSample {i+1} (Index {idx}):")
        print(f"Input Text: {original_text[:100]}...")
        
        print("Calling Real LLM...")
        try:
            llm_output = llm_agent.predict(original_text, case_explanations, "Predict weather distribution")
            print(f"LLM Output: {llm_output}")
            llm_probs = parse_llm_prediction(llm_output).to(DEVICE)
        except Exception as e:
            print(f"LLM Call Failed: {e}")
            llm_probs = torch.tensor([0.33, 0.33, 0.33]).to(DEVICE)

        # 3. Calculate Metrics
        # KL Divergence
        enc_kl = calculate_kl_divergence(encoder_probs, target)
        llm_kl = calculate_kl_divergence(llm_probs, target)
        
        # Accuracy (Highest probability category match)
        true_label = torch.argmax(target).item()
        enc_pred_label = torch.argmax(encoder_probs).item()
        llm_pred_label = torch.argmax(llm_probs).item()
        
        enc_acc = 1.0 if enc_pred_label == true_label else 0.0
        llm_acc = 1.0 if llm_pred_label == true_label else 0.0
        
        total_encoder_kl += enc_kl
        total_llm_kl += llm_kl
        total_encoder_acc += enc_acc
        total_llm_acc += llm_acc
        
        # 4. Display Results
        print("-" * 40)
        print(f"{'Metric':<15} | {'Encoder':<15} | {'LLM':<15} | {'Truth':<15}")
        print("-" * 40)
        print(f"{'No Precip':<15} | {encoder_probs[0]:.4f}          | {llm_probs[0]:.4f}          | {target[0]:.4f}")
        print(f"{'Rain':<15} | {encoder_probs[1]:.4f}          | {llm_probs[1]:.4f}          | {target[1]:.4f}")
        print(f"{'Snow':<15} | {encoder_probs[2]:.4f}          | {llm_probs[2]:.4f}          | {target[2]:.4f}")
        print("-" * 40)
        print(f"{'KL Loss':<15} | {enc_kl:.4f}          | {llm_kl:.4f}          | -")
        print(f"{'Correct?':<15} | {enc_acc == 1.0!s:<15} | {llm_acc == 1.0!s:<15} | Label: {true_label}")
        print("-" * 40)

    print("\n" + "="*80)
    print("AVERAGE RESULTS")
    print("="*80)
    print(f"Encoder - Avg KL: {total_encoder_kl/3:.4f}, Accuracy: {total_encoder_acc/3:.2%}")
    print(f"LLM     - Avg KL: {total_llm_kl/3:.4f}, Accuracy: {total_llm_acc/3:.2%}")

if __name__ == "__main__":
    main()
