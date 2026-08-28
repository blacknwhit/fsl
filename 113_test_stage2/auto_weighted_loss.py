# Automatically weighted multi-task loss
# From Mod-Squad-master/util/AutomaticWeightedLoss.py
# Based on "Multi-Task Learning Using Uncertainty to Weigh Losses"
# https://arxiv.org/abs/1705.07115

import torch
import torch.nn as nn


class AutomaticWeightedLoss(nn.Module):
    """
    Automatically weighted multi-task loss using learned uncertainty.
    
    This implements the uncertainty-based weighting from:
    "Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics"
    Kendall et al., CVPR 2018
    
    For each task loss L_i, the weighted loss is:
        weighted_L_i = 0.5 / (sigma_i^2) * L_i + log(1 + sigma_i^2)
    
    Where sigma_i is a learned parameter representing task uncertainty.
    Higher uncertainty -> lower weight on that task's loss.
    
    Params:
        num: int, the number of tasks/losses
        
    Examples:
        loss1 = 1.0
        loss2 = 2.0
        awl = AutomaticWeightedLoss(2)
        loss_sum = awl([loss1, loss2])
    """
    
    def __init__(self, num: int = 2):
        super(AutomaticWeightedLoss, self).__init__()
        # Initialize params to 1, representing equal initial uncertainty
        params = torch.ones(num, requires_grad=True)
        self.params = nn.Parameter(params)
        self.num = num

    def forward(self, losses):
        """
        Compute weighted sum of losses.
        
        Args:
            losses: list or tuple of task losses (scalars or 0-dim tensors)
            
        Returns:
            Weighted sum of losses
        """
        loss_sum = 0
        for i, loss in enumerate(losses):
            # 0.5 / sigma^2 * loss + log(1 + sigma^2)
            # The log term acts as regularization to prevent sigma from growing unbounded
            loss_sum += 0.5 / (self.params[i] ** 2) * loss + torch.log(1 + self.params[i] ** 2)
        return loss_sum
    
    def get_weights(self):
        """
        Get the current task weights (inverse of uncertainty squared).
        
        Returns:
            list of weights for each task
        """
        with torch.no_grad():
            weights = [0.5 / (self.params[i] ** 2) for i in range(self.num)]
        return weights
    
    def extra_repr(self):
        return f'num_tasks={self.num}'


if __name__ == '__main__':
    # Test
    awl = AutomaticWeightedLoss(3)
    print("Parameters:", list(awl.parameters()))
    
    loss1 = torch.tensor(1.0)
    loss2 = torch.tensor(2.0)
    loss3 = torch.tensor(0.5)
    
    total = awl([loss1, loss2, loss3])
    print(f"Total loss: {total}")
    print(f"Weights: {awl.get_weights()}")
