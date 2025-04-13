import torch
import torchvision

def unfreeze_last_n_layers(model : torch.nn.Module,
                           n : int = 0) -> torch.nn.Module:
    
    for param in model.parameters():
        param.requires_grad = False

    leaf_modules = [
        module for module in model.modules() if not list(module.children()) and list(module.parameters())  
    ]
    
    for layer in leaf_modules[::-1][:n]:
        for param in layer.parameters():
            param.requires_grad = True
    return model

def replace_classifier(model : torch.nn.Module,
                       num_classes : int,
                       layer_name : str = None) -> torch.nn.Module:
    
    if not layer_name:
        for candidate in ["classifier", "fc", "head"]:
            if hasattr(model, candidate):
                layer_name = candidate
                break
            else:
                raise ValueError("Couldn't detect classifier layer. Specify layer_name.")
    old_layer = getattr(model, layer_name)

    if isinstance(old_layer, torch.nn.Sequential):
        linear_layers = [i for i, m in enumerate(old_layer) 
                        if isinstance(m, torch.nn.Linear)]
        if not linear_layers:
            raise ValueError("No Linear layers found in sequential classifier")
        
        last_linear_idx = linear_layers[-1]
        in_features = old_layer[last_linear_idx].in_features
        
        new_layer = list(old_layer.children())
        new_layer[last_linear_idx] = torch.nn.Linear(in_features, num_classes)
        setattr(model, layer_name, torch.nn.Sequential(*new_layer))
        
    elif isinstance(old_layer, torch.nn.Linear):
        in_features = old_layer.in_features
        setattr(model, layer_name, torch.nn.Linear(in_features, num_classes))
        
    else:
        raise TypeError(f"Unsupported layer type: {type(old_layer)}")
    return model

def model_builder(model_weights: str,
        model_name: str,
        unfreeze_layers: int,
        num_classes: int,
        layer_name: str = None,
        device: str = "cpu") -> torch.nn.Module:
    
    model_weights = getattr(torchvision.models, model_weights).DEFAULT
    model_class = getattr(torchvision.models, model_name)

    model = model_class(weights= model_weights)

    model = unfreeze_last_n_layers(model= model,
                           n = unfreeze_layers)
    
    modified_model = replace_classifier(model= model,
                                        num_classes= num_classes,
                                        layer_name= layer_name)
    
    modified_model = modified_model.to(device)
    
    return modified_model