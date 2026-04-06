import numpy as np
from typing import List, Union, Dict

def calculate_metrics(pred_dist: Union[List[float], np.ndarray], 
                     true_dist: Union[List[float], np.ndarray]) -> Dict[str, float]:
    """
    Calculate KL Divergence and MSE between predicted and true distributions.
    
    Args:
        pred_dist: Predicted probability distribution (should sum to 1)
        true_dist: True probability distribution (should sum to 1)
        
    Returns:
        dict: Dictionary containing 'kl' and 'mse' values
    """
    # Convert to numpy arrays if they aren't already
    p = np.array(pred_dist, dtype=np.float64)
    q = np.array(true_dist, dtype=np.float64)
    
    # Clip values to avoid log(0)
    epsilon = 1e-10
    p = np.clip(p, epsilon, 1.0)
    q = np.clip(q, epsilon, 1.0)
    
    # Re-normalize to ensure they sum to 1 after clipping (optional but good practice)
    # p = p / p.sum()
    # q = q / q.sum()
    
    # KL Divergence: D_KL(True || Pred) = sum(True * log(True / Pred))
    kl_value = np.sum(q * np.log(q / p))
    
    # Mean Squared Error
    mse_value = np.mean((p - q) ** 2)
    
    return {
        'kl': float(kl_value),
        'mse': float(mse_value)
    }
