import torch
from pytorch_grad_cam import GradCAM

def HeatMap(model : torch.nn.Module,
            layer_name : str,
            image : torch.Tensor) -> torch.Tensor:
    """
    Create a GradCam Heatmap for the input image with respect to the model.
    
    model : torch.nn.Module 
    Layer_name : string
    image : torch.Tensor
    """
    target_layer = getattr(model, layer_name)
    cam = GradCAM(model, target_layers = [target_layer])
    heatmap = cam(image)
    return heatmap
