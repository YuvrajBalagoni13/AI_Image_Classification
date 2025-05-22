import torch
import onnx2torch
import onnx
import torchvision
from torchvision import transforms
from GradCam import gradcam
import numpy as np
import cv2

def Overlay_heatmap(image : torch.Tensor):
    
    onnx_model = onnx.load("Models/trainedmodel.onnx")
    pytorch_model = onnx2torch.convert(onnx_model)

    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])

    transform_img = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor()
    ])

    layer_name = "features/features/8/features/8/0/Conv"

    transform_image = transform(image)

    transform_image = transform_image.unsqueeze(dim = 0)
    heatmap = gradcam.HeatMap(model= pytorch_model,
                              layer_name= layer_name,
                              image= transform_image)
    
    squeeze_image = transform_image.squeeze(dim = 0)

    og_image = transform_img(image)
    np_image = og_image.numpy()

    og_image_hwc = np.transpose(np_image,(1,2,0))
    heatmap_hwc = np.transpose(heatmap, (1,2,0))
    heatmap_2d = heatmap_hwc[:, :, 0]

    heatmap_norm = (heatmap_2d - heatmap_2d.min())/(heatmap_2d.max() - heatmap_2d.min())
    heatmap_uint8 = np.uint8(heatmap_norm * 255)

    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color = cv2.applyColorMap(heatmap_color, cv2.COLOR_BGR2RGB)

    if og_image_hwc.max() <= 1.0:
        img_rgb_uint8 = np.uint8(255 * og_image_hwc)
    else:
        img_rgb_uint8 = og_image_hwc.astype(np.uint8)

    alpha = 0.25

    overlay = cv2.addWeighted(img_rgb_uint8, 1 - alpha, heatmap_color, alpha, 0)
    return overlay