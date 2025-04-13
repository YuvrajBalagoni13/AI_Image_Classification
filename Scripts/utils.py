import torch
from typing import Tuple

def create_loss_and_optim(model: torch.nn.Module, 
                          unfreeze_layers: int, 
                          learning_rate_classifier: float, 
                          learning_rate_unfreeze: float) -> Tuple[torch.nn.Module, torch.optim.Optimizer]:

    leaf_modules = [
        module for module in model.modules() if not list(module.children()) and list(module.parameters())
    ]
    data = []
    data1 = []

    for layer in leaf_modules[::-1][1: unfreeze_layers]:
        for param in layer.parameters():
            data.append(param)

    for param in leaf_modules[::-1][0].parameters():
            data1.append(param)

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam([
         {"params": data1, "lr": learning_rate_classifier},
         {"params": data, "lr": learning_rate_unfreeze}
    ], weight_decay= 1e-4)

    return loss_fn, optimizer