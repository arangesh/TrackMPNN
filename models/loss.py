import torch
import torch.nn as nn

class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float, int)): self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average
        self.eps = 1e-10

    def forward(self, output, target):
        output = torch.stack((1 - output, output), dim=1) # (N, 2)
        target = target.view(-1, 1) # (N, 1)

        logpt = torch.log(output + self.eps)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = torch.exp(logpt)

        if self.alpha is not None:
            if self.alpha.type() != output.data.type():
                self.alpha = self.alpha.type_as(output.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * at

        loss = -1 * (1 - pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()