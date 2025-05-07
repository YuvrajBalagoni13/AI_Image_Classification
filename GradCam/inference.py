import onnxruntime as ort
import torch
import torchvision
from torchvision import transforms
import numpy as np

def inference(model_path : str = "Models/Modelshybridmodel.onnx",
              image : torch.Tensor = None,
              device : torch.device = None) -> {str , int}:
    
    class_names = ["FAKE", "REAL"]

    transform = transforms.Compose([
        transforms.Resize((32,32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])

    transformed_image = transform(image)
    x = transformed_image.unsqueeze(dim = 0)

    ort_sess = ort.InferenceSession(model_path)
    out_ort = ort_sess.run(None, {"input": x.numpy()})

    output = class_names[np.argmax(out_ort)]

    confidence = np.max(out_ort[0]) / 10

    return output, confidence
