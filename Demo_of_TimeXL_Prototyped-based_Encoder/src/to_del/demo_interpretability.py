import torch
import torch.nn.functional as F
import numpy as np
import logging
import os
import sys
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_ROOT)

from training.models import TimeXLModel, PrototypeManager
from .real_data_loader import ProcessedWeatherDataset
from training.base_trainer import BaseTrainer
from training.loss import TimeXLLoss

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("InterpretabilityDemo")

def demonstrate_interpretability():
    logger.info("="*60)
    logger.info("       TIMEXL INTERPRETABILITY DEMO       ")
    logger.info("="*60)
    
    # Configuration
    DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
    CITY = 'San Francisco'
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 1. Load Data
    logger.info(f"Loading data for {CITY}...")
    # We need Training Set to build the explanation library (prototypes map to training samples)
    train_dataset = ProcessedWeatherDataset(DATA_DIR, city=CITY, split='train')
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)
    
    # We pick one Test Sample to explain
    test_dataset = ProcessedWeatherDataset(DATA_DIR, city=CITY, split='test')
    test_sample_idx = 10 # Sample with mixed weather (Rain/Clouds)
    x_time, original_text, target = test_dataset[test_sample_idx]
    
    # 2. Load Model
    logger.info("Loading trained model...")
    num_classes = 3 # No Precip, Rain, Snow
    k = 10 
    model = TimeXLModel(num_classes=num_classes, k=k).to(DEVICE)
    pm = PrototypeManager(num_classes=num_classes, k=k).to(DEVICE)
    
    try:
        model.load_state_dict(torch.load(os.path.join(PROJECT_ROOT, 'real_data_checkpoints', 'best_model.pth'), map_location=DEVICE))
        # pm.load_state_dict(torch.load(os.path.join(PROJECT_ROOT, 'prototypes.cpython-312.pth'), map_location=DEVICE))
    except FileNotFoundError:
        logger.error("Model checkpoints not found!")
        return

    model.eval()
    
    # 3. Build Explanation Library (Project Prototypes to Raw Data)
    # Usually this is done once after training. We do it here on the fly.
    # To save time, we'll limit the loader to first N batches if dataset is huge, 
    # but for best quality we use full train set.
    logger.info("Building Explanation Library (mapping prototypes to raw training data)...")
    
    # Create a temporary trainer just to use the projection method
    loss_fn = TimeXLLoss()
    optimizer = torch.optim.Adam(model.parameters()) # Dummy
    trainer = BaseTrainer(model, pm, loss_fn, optimizer, device=DEVICE)
    
    # We limit train_loader for speed in this demo, but ideally use full
    # For demo, let's just use the first 500 samples
    subset_indices = range(500)
    train_subset = torch.utils.data.Subset(train_dataset, subset_indices)
    train_loader_subset = DataLoader(train_subset, batch_size=64, shuffle=False)
    
    time_expl_lib, text_expl_lib, label_expl_lib = trainer.project_prototypes_to_raw_data(train_loader_subset)
    
    # 4. Explain the Test Sample
    logger.info("\n" + "-"*60)
    logger.info(f"EXPLAINING TEST SAMPLE (Index {test_sample_idx})")
    logger.info("-"*60)
    
    # Forward Pass
    x_time_batch = x_time.unsqueeze(0).to(DEVICE)
    z_time = model.time_encoder(x_time_batch)
    z_text = model.text_encoder([original_text])
    
    # Get Similarities
    # sim_text shape: [1, C*K] (Flattened)
    # We need to reshape back to [1, C, K] to find top activations
    sim_time_flat, sim_text_flat = pm(z_time, z_text)
    
    sim_time = sim_time_flat.reshape(1, num_classes, k)
    sim_text = sim_text_flat.reshape(1, num_classes, k)
    
    # Fusion Prediction
    logits = model.fusion(sim_time_flat, sim_text_flat)
    probs = F.softmax(logits, dim=1).squeeze(0)
    
    predicted_class = torch.argmax(probs).item()
    class_names = ['No Precipitation', 'Rain', 'Snow']
    pred_name = class_names[predicted_class]
    
    logger.info(f"Input Text: {original_text[:100]}...")
    logger.info(f"True Label Distribution: {target.tolist()}")
    logger.info(f"Model Prediction: {probs.tolist()} -> {pred_name}")
    
    # 5. Find Most Influential Prototypes
    # We look at the prototypes for the PREDICTED class that had the highest similarity
    # similarity is computed as dot product (or distance based in PM forward?)
    # Wait, PM forward uses einsum: z * P -> similarity (dot product).
    # Higher value = more similar.
    
    top_n = 3
    logger.info(f"\nWhy did the model predict '{pred_name}'?")
    logger.info(f"Top {top_n} Most Similar Historical Cases (Prototypes) for class '{pred_name}':")
    
    # Get similarities for the predicted class
    # sim_text[0, predicted_class, :] is a vector of size K
    class_sims = sim_text[0, predicted_class, :]
    
    # Get indices of top K similarities
    top_k_vals, top_k_indices = torch.topk(class_sims, top_n)
    
    for rank, (val, proto_idx) in enumerate(zip(top_k_vals, top_k_indices)):
        proto_idx = proto_idx.item()
        key = (predicted_class, proto_idx)
        
        if key in text_expl_lib:
            raw_text = text_expl_lib[key]
            raw_label = label_expl_lib[key]
            
            logger.info(f"\n  Reason #{rank+1} (Similarity: {val:.4f}):")
            logger.info(f"  Matches historical case: \"{raw_text[:80]}...\"")
            logger.info(f"  Historical Outcome: {raw_label}")
        else:
            logger.warning(f"  Prototype {key} not found in library (maybe due to subset loading).")

    # 6. Counterfactual / Contrastive Explanation (Optional)
    # "Why not Rain?" (if predicted No Precip)
    # Find best match in the alternative class
    
    other_classes = [c for c in range(num_classes) if c != predicted_class]
    if other_classes:
        alt_class = other_classes[0] # Pick first alternative (e.g., Rain)
        alt_name = class_names[alt_class]
        
        logger.info(f"\nWhy NOT '{alt_name}'?")
        alt_sims = sim_text[0, alt_class, :]
        best_alt_val, best_alt_idx = torch.max(alt_sims, dim=0)
        best_alt_idx = best_alt_idx.item()
        
        key = (alt_class, best_alt_idx)
        if key in text_expl_lib:
            raw_text = text_expl_lib[key]
            logger.info(f"  Closest evidence for '{alt_name}' was (Similarity: {best_alt_val:.4f}):")
            logger.info(f"  \"{raw_text[:80]}...\"")
            logger.info(f"  But this similarity was lower than the evidence for '{pred_name}'.")

if __name__ == "__main__":
    demonstrate_interpretability()
