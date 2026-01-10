import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    Implementação da Focal Loss para problemas de classificação desbalanceada.
    
    A Focal Loss adiciona um fator (1 - pt)^gamma à Cross Entropy padrão.
    Isso reduz a perda relativa para exemplos bem classificados (pt > 0.5) e 
    coloca mais foco em exemplos difíceis/mal classificados.
    
    Args:
        alpha (float, optional): Peso para a classe positiva (ou lista de pesos).
                                 Se for uma lista, deve ter tamanho igual a num_classes.
        gamma (float): Fator de foco. Valores maiores (e.g. 2.0) reduzem mais a perda de exemplos fáceis.
        reduction (str): 'mean' ou 'sum'.
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        
        # Trata alpha para ser um tensor se for fornecido
        if isinstance(alpha, (float, int)):
            self.alpha = torch.tensor([alpha, 1-alpha])
        elif isinstance(alpha, list):
            self.alpha = torch.tensor(alpha)
        elif isinstance(alpha, torch.Tensor):
            self.alpha = alpha
        else:
            self.alpha = None

    def forward(self, inputs, targets):
        """
        inputs: (N, C) logits do modelo (antes da softmax/sigmoid)
        targets: (N) índices das classes reais
        """
        # Cross Entropy Loss padrão (log_softmax + nll_loss)
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        
        # Calcula pt (probabilidade da classe correta)
        pt = torch.exp(-ce_loss)
        
        # Termo de foco: (1 - pt)^gamma
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
