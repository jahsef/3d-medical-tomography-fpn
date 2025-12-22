import torch
import torch.nn as nn
import torch.nn.functional as F


# class PeakBCELoss(nn.Module):
#     def __init__(self, epsilon:float, lambda_param:float, reduction = 'mean'):
#         """PeakBCELoss for continuous regression targets [0,1]. can focus on peaks or be used without peak weighting.

#         Args:
#             epsilon (float) : if target is all 0s, then this is the total weight. so its roughly epsilon vs peak weight (about 15-40 for 10x18x18 sigma about 1.3 (read patchtomodataset.py for more info on how gaussian is computed))
#             lambda_param (float): [1,inf), 1 places no emphasis on peaks, anything above 1 weights peaks higher

#             reduction (str, optional): _description_. Defaults to 'mean'.
#         """
#         super().__init__()
#         self.epsilon = epsilon
#         self.lambda_param = lambda_param
#         self.reduction = reduction

#     def forward(self, inputs, targets):
#         extra_peak_weighting = targets ** self.lambda_param
#         background_suppression_weighting = self.epsilon + targets #small base weight for background regardless
#         weights = background_suppression_weighting + extra_peak_weighting
#         #normalizing epsilon by patch.numel() makes it patch size invariant
#         #so total background weight is now epsilon * total_peak_weight
#         #for example when eps = 0.1, total_background_weight = 0.1 * total_peak_weight

#         #centernet style: y^lambda + (1 - y)^ beta (explicitly tunable background weighting but adds weird nonlinear hyperparam)
#         bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
#         weighted_loss =  bce_loss * weights
#     #     print(f"Weight stats: min={weights.min():.6f}, max={weights.max():.6f}, "
#     #   f"mean={weights.mean():.6f}, peak_sum={total_peak_weight:.2f}")
#         return weighted_loss.mean() if self.reduction == 'mean' else weighted_loss

class WeightedBCELoss(nn.Module):
    def __init__(self, pos_weight=10.0, reduction = 'mean'):
        super().__init__()
        self.pos_weight = pos_weight
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Weight based on target values - higher targets get more weight
        weights = 1.0 + (targets * self.pos_weight)
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        weighted_loss = bce_loss * weights

        return weighted_loss.mean() if self.reduction == 'mean' else weighted_loss

class BCEFocalLoss(nn.Module):
    def __init__(self, gamma=1.0):
        super().__init__()
        self.gamma = gamma
    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pred_prob = F.sigmoid(inputs)
        focal_loss = torch.abs(targets-pred_prob)**self.gamma * bce_loss
        return focal_loss.mean()

# class MAEFocalLoss(nn.Module):
#     def __init__(self, gamma=1.0):
#         super().__init__()
#         self.gamma = gamma
#     def forward(self, inputs, targets):
#         mae_loss = F.l1_loss(inputs, targets, reduction='none')
#         pred_prob = F.sigmoid(inputs)
#         focal_loss = torch.abs(targets-pred_prob)**self.gamma * mae_loss
#         return focal_loss.mean()

# class MSEFocalLoss(nn.Module):
#     def __init__(self, gamma=1.0):
#         super().__init__()
#         self.gamma = gamma
#     def forward(self, inputs, targets):
#         mse_loss = F.mse_loss(inputs, targets, reduction='none')
#         pred_prob = F.sigmoid(inputs)
#         focal_loss = torch.abs(targets-pred_prob)**self.gamma * mse_loss
#         return focal_loss.mean()

class AdaptedCornerNetLoss(nn.Module):
    def __init__(self,pos_threshold, alpha=2.0, beta=4.0):
        """CornerNet loss adapted for heatmap regression.
        
        Args:
            alpha: focal power for hard examples (default 2)
            beta: gaussian penalty reduction power (default 4)
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.pos_threshold = pos_threshold

    def forward(self, inputs, targets):
        """
        Args:
            inputs: predicted logits [B, C, H, W]
            targets: ground truth heatmaps with gaussian bumps [B, C, H, W]
        """
        pred = torch.sigmoid(inputs)
        #need dynamic epsilon based on numerical precision, otherwise can become 1.0 or 0.0 which in log(1-p) or log(p) results in inf which causes explosions
        if pred.dtype == torch.float16:
            eps = 1e-3
        elif pred.dtype == torch.bfloat16:
            eps = 1e-2
        else:  # float32 or float64
            eps = 1e-6
        pred = torch.clamp(pred, min=eps, max=1 - eps)

        # Positive loss: -(t-p)^α * log(p) 
        # Applied where y >= pos_threshold
        pos_loss = -((1 - pred) ** self.alpha) * torch.log(pred)
        # Negative loss: -(1-y)^β * p^α * log(1-p)
        # Applied where y < 1 (background and gaussian falloff)
        # (1-y)^β reduces penalty near peaks (gaussian bumps)
        neg_loss = -((1 - targets) ** self.beta) * (pred ** self.alpha) * torch.log(1 - pred)

        loss = torch.where(targets >= self.pos_threshold, pos_loss, neg_loss)
        return loss.mean()

class FuzzyCornerNetLoss(nn.Module):
    def __init__(self,pos_threshold, alpha=2.0, beta=4.0):
        """CornerNet-style focal loss for heatmap regression.

        Args:
            alpha: focal power for hard examples (default 2)
            beta: gaussian penalty reduction power (default 4)
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.pos_threshold = pos_threshold

    def forward(self, inputs, targets):
        """
        Args:
            inputs: predicted logits [B, C, H, W]
            targets: ground truth heatmaps with gaussian bumps [B, C, H, W]
        """
        pred = torch.sigmoid(inputs)
        #need dynamic epsilon based on numerical precision, otherwise can become 1.0 or 0.0 which in log(1-p) or log(p) results in inf which causes explosions
        if pred.dtype == torch.float16:
            eps = 1e-3
        elif pred.dtype == torch.bfloat16:
            eps = 1e-2
        else:  # float32 or float64
            eps = 1e-6
        pred = torch.clamp(pred, min=eps, max=1 - eps)

        #TARGETS - PRED INSTEAD OF 1-PRED
        pos_loss = -((targets - pred) ** self.alpha) * torch.log(pred)
        
        #proposed = -((targets - pred) ** self.alpha) * (pred ** self.alpha) * torch.log(pred)

        neg_loss = -((1 - targets) ** self.beta) * (pred ** self.alpha) * torch.log(1 - pred)

        loss = torch.where(targets >= self.pos_threshold, pos_loss, neg_loss)
        #modified to be 0.75 to allow subgrid precision stuff (realspace => ds pixel if close enough to the edge, gaussian bleeds over enough and for pos sample we consider 0.75)

        return loss.mean()
    
class RegressionCornerNetLoss(nn.Module):
    def __init__(self,pos_threshold, alpha=2.0, beta=4.0):
        """ADAPTS POS CASE TO BE MORE TRUE REGRESSION

        Args:
            alpha: focal power for hard examples (default 2)
            beta: gaussian penalty reduction power (default 4)
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.pos_threshold = pos_threshold

    def forward(self, inputs, targets):
        """
        Args:
            inputs: predicted logits [B, C, H, W]
            targets: ground truth heatmaps with gaussian bumps [B, C, H, W]
        """
        pred = torch.sigmoid(inputs)
        #need dynamic epsilon based on numerical precision, otherwise can become 1.0 or 0.0 which in log(1-p) or log(p) results in inf which causes explosions
        if pred.dtype == torch.float16:
            eps = 1e-3
        elif pred.dtype == torch.bfloat16:
            eps = 1e-2
        else:  # float32 or float64
            eps = 1e-6
        pred = torch.clamp(pred, min=eps, max=1 - eps)
        
        #TARGETS - PRED INSTEAD OF 1-PRED
        error = torch.abs(targets - pred)
        pos_loss = -(error ** self.alpha) * torch.log(1-error)
        
        #proposed = -((targets - pred) ** self.alpha) * (pred ** self.alpha) * torch.log(pred)

        neg_loss = -((1 - targets) ** self.beta) * (pred ** self.alpha) * torch.log(1 - pred)

        loss = torch.where(targets >= self.pos_threshold, pos_loss, neg_loss)
        #modified to be 0.75 to allow subgrid precision stuff (realspace => ds pixel if close enough to the edge, gaussian bleeds over enough and for pos sample we consider 0.75)

        return loss.mean()
    
class CombinedFocalLoss(nn.Module):
    def __init__(self, alpha=2.0, beta=4.0):
        """Generalized BCE with focal and gaussian falloff weighting.

        At alpha=1, beta=1 this is essentially BCE.

        Args:
            alpha: error tolerance - higher ignores small errors (focal power)
            beta: falloff tolerance - higher ignores mid conf regions (gaussian falloff)
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, inputs, targets):
        pred = torch.sigmoid(inputs)

        if pred.dtype == torch.float16:
            eps = 1e-3
        elif pred.dtype == torch.bfloat16:
            eps = 1e-2
        else:
            eps = 1e-6
        pred = torch.clamp(pred, min=eps, max=1 - eps)

        # Focal weighting based on error
        focal_weighting = torch.abs(targets - pred) ** self.alpha
        # Pos term: target-gated, focal-weighted, push toward higher pred
        pos_loss = (targets ** self.beta) * focal_weighting * torch.log(pred)

        # Neg term: background-gated, high-conf FP crushing
        high_conf_fp_weighting = pred ** self.alpha
        neg_loss = ((1 - targets) ** self.beta) * high_conf_fp_weighting * torch.log(1 - pred)

        loss = -(pos_loss + neg_loss)
        return loss.mean()
    
