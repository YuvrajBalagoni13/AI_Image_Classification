import onnxruntime as ort
import numpy as np
from PIL import Image

def inference(model_path = "Models/finalmodel.onnx",
              image  = None) -> {str , int}:
    
    class_names = ["FAKE", "REAL"]

    img = image.resize((32, 32))
    img = np.asarray(img, dtype=np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = (img - mean) / std
    transformed_image = np.transpose(img, (2, 0, 1))

    x = np.expand_dims(transformed_image, axis=0)
    x = x.astype(np.float32)

    ort_sess = ort.InferenceSession(model_path)
    out_ort = ort_sess.run(None, {"input": x})

    output = class_names[np.argmax(out_ort)]

    confidence = np.max(out_ort[0]) / 10

    return output, confidence
