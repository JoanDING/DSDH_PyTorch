import torch.nn as nn
import torch

import pdb

class DSDHLoss(nn.Module):
    def __init__(self, eta):
        super(DSDHLoss, self).__init__()
        self.eta = eta

    def forward(self, U_batch, U, S, B):
        theta = U.t() @ U_batch / 2

        # Prevent exp overflow
        theta = torch.clamp(theta, min=-100, max=50)

        metric_loss = (torch.log(1 + torch.exp(theta)) - S * theta).mean()
        quantization_loss = (B - U_batch).pow(2).mean()
        loss = metric_loss + self.eta * quantization_loss

        return loss

class DDDHLoss(nn.Module):
    def __init__(self, nu, eta):
        super(DDDHLoss, self).__init__()
        self.eta = eta
        self.nu = nu
        #self.l2_loss = torch.nn.MSELoss()

    def forward(self, f1, f2, S, y1, y2, g1, g2):
        # f1, f2: output of hash code layer before binarization
        # g1, g1: output of classification layer, bs*label_len
        # y1, y2: labels of samples, bs*code_len
        # S: similarity matrix
        S = S.float()
        theta = f1 @ f2.t() / 2
        sim = S.size()[0] * S.size()[1]
        sim1 = torch.sum(S)
        sim0 = sim - sim1
        balance_weight1 = sim / sim1.float()
        balance_weight0 = sim / sim0.float()

        balance_s1 = balance_weight1*S
        balance_s0 = balance_weight0*(S-1)
        balance_s = balance_s1 - balance_s0

        metric_loss = (balance_s * (torch.log(1 + torch.exp(theta)) - S * theta)).mean()

        classification_loss = (torch.norm(y1-g1) + torch.norm(y2-g2))/(2*y1.size()[-1])
        #classification_loss2 = self.l2_loss(y1,g1) + self.l2_loss(y2,g2)
        quantization_loss = (torch.norm(torch.sign(f1) - f1) + torch.norm(torch.sign(f2)-f2))/(2*f1.size()[-1])
        #quantization_loss = self.l2_loss(torch.sign(f1), f1) + self.l2_loss(torch.sign(f2), f2)

        loss = metric_loss + self.nu*classification_loss + self.eta*quantization_loss

        return loss
