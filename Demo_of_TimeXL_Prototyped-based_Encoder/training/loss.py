import torch
import torch.nn as nn
import torch.nn.functional as F

class TimeXLLoss(nn.Module):
    def __init__(self, lambda_1=0.01, lambda_2=0.01, lambda_3=0.01, d_min=2.0):
        """
        TimeXL Joint Loss Function
        
        Loss = L_KL + lambda_1 * L_Clst + lambda_2 * L_Evi + lambda_3 * L_Div
        """
        super().__init__()
        self.lambda_1 = lambda_1 # Clustering
        self.lambda_2 = lambda_2 # Evidence
        self.lambda_3 = lambda_3 # Diversity
        self.d_min = d_min
        # Change: Use KLDivLoss for soft labels (probability distributions)
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')

    def forward(self, pred, target, z_time, z_text, pm):
        """
        Args:
            pred: [B, num_classes] - Classification logits (unnormalized)
            target: [B, num_classes] - Target probability distribution (soft labels)
            z_time: [B, time_dim] - Time embeddings
            z_text: [B, text_dim] - Text embeddings
            pm: PrototypeManager containing P_time [C, K, Dt] and P_text [C, K, Dx]
        """
        
        # 1. KL Divergence Loss
        # KLDivLoss expects input to be log-probabilities
        pred_log_probs = F.log_softmax(pred, dim=1)
        L_kl = self.kl_loss(pred_log_probs, target)
        
        # For prototype losses, we need to assign each sample to a "dominant" class
        # to decide which prototypes it should be close to.
        # We use the class with the highest probability in the target distribution.
        target_indices = torch.argmax(target, dim=1) # [B]
        
        # Helper to compute squared euclidean distances
        def compute_dist(z, P):
            # z: [B, D]
            # P: [C, K, D]
            # Output: [B, C, K]
            
            # Expand z: [B, 1, 1, D]
            z_exp = z.unsqueeze(1).unsqueeze(1)
            # Expand P: [1, C, K, D]
            P_exp = P.unsqueeze(0)
            
            # Dist: ||z - p||^2
            dist = torch.sum((z_exp - P_exp) ** 2, dim=-1)
            return dist

        # Get prototypes
        P_time = pm.P_time # [C, K, Dt]
        P_text = pm.P_text # [C, K, Dx]
        
        # Compute distances for all samples to all prototypes
        # dist_time: [B, C, K]
        dist_time = compute_dist(z_time, P_time)
        dist_text = compute_dist(z_text, P_text)
        
        batch_size = target.size(0)
        num_classes = P_time.size(0)
        k = P_time.size(1)
        
        # Gather distances corresponding to the dominant class for each sample
        # target_indices: [B] -> expand to [B, 1, K] for gathering
        target_exp = target_indices.view(batch_size, 1, 1).expand(-1, 1, k)
        
        # dist_time_c: [B, 1, K] -> [B, K]
        # This gives distance of sample i to all K prototypes of its dominant class
        dist_time_c = torch.gather(dist_time, 1, target_exp).squeeze(1)
        dist_text_c = torch.gather(dist_text, 1, target_exp).squeeze(1)
        
        # ----------------------------------------------------------------
        # 2. Clustering Loss (L_Clst)
        # Force sample to be close to *nearest* prototype of its class
        # min_{j} ||z_i - p_{c,j}||^2
        # ----------------------------------------------------------------
        
        # min over K prototypes
        min_dist_time, _ = torch.min(dist_time_c, dim=1) # [B]
        min_dist_text, _ = torch.min(dist_text_c, dim=1) # [B]
        
        L_clst = torch.mean(min_dist_time) + torch.mean(min_dist_text)
        
        # ----------------------------------------------------------------
        # 3. Evidence Loss (L_Evi)
        # Force each prototype to have at least one sample close to it
        # min_{i \in Batch_c} ||z_i - p_{c,j}||^2
        # ----------------------------------------------------------------
        
        L_evi_time = 0.0
        L_evi_text = 0.0
        
        present_classes = torch.unique(target_indices)
        
        for c in present_classes:
            # Mask for samples belonging to class c (dominant class)
            mask = (target_indices == c)
            
            # Get distances for these samples to prototypes of class c
            # dist_time_c[mask] gives [N_c, K]
            d_time_samples = dist_time_c[mask]
            d_text_samples = dist_text_c[mask]
            
            if d_time_samples.size(0) == 0:
                continue
                
            # For each prototype k, find min distance among samples
            # min over samples (dim 0)
            min_d_time, _ = torch.min(d_time_samples, dim=0) # [K]
            min_d_text, _ = torch.min(d_text_samples, dim=0) # [K]
            
            L_evi_time += torch.sum(min_d_time)
            L_evi_text += torch.sum(min_d_text)
            
        L_evi = L_evi_time + L_evi_text
        
        # ----------------------------------------------------------------
        # 4. Diversity Loss (L_Div)
        # Prototypes within same class should be far apart
        # sum max(0, d_min - ||p_a - p_b||^2)
        # ----------------------------------------------------------------
        
        def compute_div(P):
            # P: [C, K, D]
            loss = 0.0
            for c in range(num_classes):
                p_c = P[c] # [K, D]
                # Pairwise distances [K, K]
                dists = torch.cdist(p_c, p_c, p=2) 
                dists_sq = dists ** 2
                
                # Upper triangle only (exclude diagonal and duplicates)
                triu_idx = torch.triu_indices(k, k, offset=1)
                dists_sq_upper = dists_sq[triu_idx[0], triu_idx[1]]
                
                # Hinge loss
                l = torch.relu(self.d_min - dists_sq_upper)
                loss += torch.sum(l)
            return loss
            
        L_div_time = compute_div(P_time)
        L_div_text = compute_div(P_text)
        L_div = L_div_time + L_div_text
        
        # Total Loss
        total_loss = L_kl + self.lambda_1 * L_clst + self.lambda_2 * L_evi + self.lambda_3 * L_div
        
        return total_loss, {
            'loss': total_loss.item(),
            'kl': L_kl.item(),
            'clst': L_clst.item(),
            'evi': L_evi.item(),
            'div': L_div.item()
        }
